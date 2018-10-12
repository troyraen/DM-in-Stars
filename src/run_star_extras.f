! ***********************************************************************
!
!   Copyright (C) 2011  Bill Paxton
!
!   this file is part of mesa.
!
!   mesa is free software; you can redistribute it and/or modify
!   it under the terms of the gnu general library public license as published
!   by the free software foundation; either version 2 of the license, or
!   (at your option) any later version.
!
!   mesa is distributed in the hope that it will be useful,
!   but without any warranty; without even the implied warranty of
!   merchantability or fitness for a particular purpose.  see the
!   gnu library general public license for more details.
!
!   you should have received a copy of the gnu library general public license
!   along with this software; if not, write to the free software
!   foundation, inc., 59 temple place, suite 330, boston, ma 02111-1307 usa
!
! ***********************************************************************

      module run_star_extras


      use crlibm_lib ! MIST
      use rates_def ! MIST
      use net_def ! MIST
      use star_lib
      use star_def
      use const_def
      use chem_def
      use wimp_module   ! necessary to point towards the other_energy hook

      implicit none

      integer :: time0, time1, clock_rate
      double precision, parameter :: expected_runtime = 16.5 ! minutes
      real(dp) :: original_diffusion_dt_limit ! MIST
      real(dp) :: burn_check = 0.0 ! MIST


      ! these routines are called by the standard run_star check_model
      contains

      subroutine extras_controls(id, ierr)
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return

         s% extras_startup => extras_startup
         s% extras_check_model => extras_check_model
         s% extras_finish_step => extras_finish_step
         s% extras_after_evolve => extras_after_evolve
         s% how_many_extra_history_columns => how_many_extra_history_columns
         s% data_for_extra_history_columns => data_for_extra_history_columns
         s% how_many_extra_profile_columns => how_many_extra_profile_columns
         s% data_for_extra_profile_columns => data_for_extra_profile_columns

         s% other_energy_implicit => wimp_energy_transport ! subroutine where extra_heat is defined inside of module wimp_module

         original_diffusion_dt_limit = s% diffusion_dt_limit ! MIST
         s% other_wind => Reimers_then_Blocker ! MIST

      end subroutine extras_controls

!!! MIST
    subroutine Reimers_then_Blocker(id, Lsurf, Msurf, Rsurf, Tsurf, w, ierr)
    use star_def
    use chem_def, only: ih1, ihe4
    integer, intent(in) :: id
    real(dp), intent(in) :: Lsurf, Msurf, Rsurf, Tsurf ! surface values (cgs)
    !     NOTE: surface is outermost cell. not necessarily at photosphere.
    !     NOTE: don't assume that vars are set at this point.
    !     so if you want values other than those given as args,
    !     you should use values from s% xh(:,:) and s% xa(:,:) only.
    !     rather than things like s% Teff or s% lnT(:) which have not been set yet.
    real(dp), intent(out) :: w ! wind in units of Msun/year (value is >= 0)
    integer, intent(out) :: ierr
    integer :: h1, he4
    real(dp) :: plain_reimers, reimers_w, blocker_w, center_h1, center_he4
    type (star_info), pointer :: s
    ierr = 0
    call star_ptr(id, s, ierr)
    if (ierr /= 0) return

    plain_reimers = 4d-13*(Lsurf*Rsurf/Msurf)/(Lsun*Rsun/Msun)

    reimers_w = plain_reimers * s% Reimers_scaling_factor
    blocker_w = plain_reimers * s% Blocker_scaling_factor * &
           4.83d-9 * pow_cr(Msurf/Msun,-2.1d0) * pow_cr(Lsurf/Lsun,2.7d0)

      h1 = s% net_iso(ih1)
      he4 = s% net_iso(ihe4)
      center_h1 = s% xa(h1,s% nz)
      center_he4 = s% xa(he4,s% nz)

      !prevent the low mass RGBs from using Blocker
      if (center_h1 < 0.01d0 .and. center_he4 > 0.1d0) then
         w = reimers_w
      else
         w = max(reimers_w, blocker_w)
      end if

    end subroutine Reimers_then_Blocker
!!! MIST


      integer function extras_startup(id, restart, ierr)
         integer, intent(in) :: id
         logical, intent(in) :: restart
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         extras_startup = 0
         call system_clock(time0,clock_rate)
         if (.not. restart) then
            call alloc_extra_info(s)
         else ! it is a restart
            call unpack_extra_info(s)
         end if
      end function extras_startup


      subroutine extras_after_evolve(id, id_extra, ierr)
         integer, intent(in) :: id, id_extra
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         double precision :: dt
         character (len=strlen) :: test
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         call system_clock(time1,clock_rate)
         dt = dble(time1 - time0) / clock_rate / 60
         call GET_ENVIRONMENT_VARIABLE( &
            "MESA_TEST_SUITE_CHECK_RUNTIME", test, status=ierr, trim_name=.true.)
         if (ierr == 0 .and. trim(test) == 'true' .and. dt > 1.5*expected_runtime) then
            write(*,'(/,a70,2f12.1,99i10/)') &
               'failed: EXCESSIVE runtime, prev time, retries, backups, steps', &
               dt, expected_runtime, s% num_retries, s% num_backups, s% model_number
         else
            write(*,'(/,a50,2f12.1,99i10/)') 'runtime, prev time, retries, backups, steps', &
               dt, expected_runtime, s% num_retries, s% num_backups, s% model_number
         end if
         ierr = 0
      end subroutine extras_after_evolve


      ! returns either keep_going, retry, backup, or terminate.
      integer function extras_check_model(id, id_extra)
         integer, intent(in) :: id, id_extra
         integer :: ierr, r, burn_category ! MIST
         real(dp) :: envelope_mass_fraction, L_He, L_tot, orig_eta, target_eta, min_center_h1_for_diff, critmass, feh ! MIST
         real(dp) :: category_factors(num_categories) ! MIST
         real(dp), parameter :: huge_dt_limit = 3.15d16 ! ~1 Gyr ! MIST
         real(dp), parameter :: new_varcontrol_target = 1d-3 ! MIST
         real(dp), parameter :: Zsol = 0.0142 ! MIST
         type (Net_General_Info), pointer :: g ! MIST
         character (len=strlen) :: photoname ! MIST
         type (star_info), pointer :: s

         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         extras_check_model = keep_going

         ierr = 0 ! MIST
         call get_net_ptr(s% net_handle, g, ierr) ! MIST
         if (ierr /= 0) stop 'bad handle' ! MIST

!!!! MIST
!     increase VARCONTROL and MDOT: increase varcontrol and Mdot when the model hits the TPAGB phase
     if ((s% initial_mass < 10) .and. (s% center_h1 < 1d-4) .and. (s% center_he4 < 1d-4)) then
        !try turning up Mdot
        feh = log10_cr((1.0 - (s% job% initial_h1 + s% job% initial_h2 + s% job% initial_he3 + s% job% initial_he4))/Zsol)
        if (feh < -0.3) then
           critmass = pow_cr(feh,2d0)*0.3618377 + feh*1.47045658 + 5.69083898
           if (feh < -2.15) then
              critmass = pow_cr(-2.15d0,2d0)*0.3618377 -2.15*1.47045658 + 5.69083898
           end if
        else if ((feh >= -0.3) .and. (feh <= -0.22)) then
           critmass = feh*18.75 + 10.925
        else
           critmass = feh*1.09595794 + 7.0660861
        end if
        if ((s% initial_mass > critmass) .and. (s% have_done_TP)) then
           if (s% Blocker_scaling_factor < 1.0) then
              write(*,*) '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
              write(*,*) 'turning up Blocker'
              write(*,*) '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
           end if
           s% Blocker_scaling_factor = 3.0
        end if

        if ((s% have_done_TP) .and. (s% varcontrol_target < new_varcontrol_target)) then !only print the first time
           s% varcontrol_target = new_varcontrol_target

!     CONVERGENCE TEST CHANGING C
    s% varcontrol_target = s% varcontrol_target * 1.0

           write(*,*) '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
           write(*,*) 'increasing varcontrol to ', s% varcontrol_target
           write(*,*) '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
        end if
     end if

!     treat postAGB: suppress late burning by turn off burning post-AGB and also save a model and photo
     envelope_mass_fraction = 1d0 - max(s% he_core_mass, s% c_core_mass, s% o_core_mass)/s% star_mass
     category_factors(:) = 1.0 !turn off burning except for H
     category_factors(3:) = 0.0
     if ((s% initial_mass < 10) .and. (envelope_mass_fraction < 0.1) .and. (s% center_h1 < 1d-4) .and. (s% center_he4 < 1d-4) &
     .and. (s% L_phot > 3.0) .and. (s% Teff > 7000.0)) then
		  if (burn_check == 0.0) then !only print the first time
			  write(*,*) '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
			  write(*,*) 'now at post AGB phase, turning off all burning except for H & saving a model + photo'
			  write(*,*) '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'

			  !save a model and photo
			  call star_write_model(id, s% job% save_model_filename, ierr)
			  photoname = 'photos/pAGB_photo'
			  call star_save_for_restart(id, photoname, ierr)

			  !turn off burning
			  do r=1,g% num_reactions
				  burn_category = reaction_categories(g% reaction_id(r))
				  s% rate_factors(r) = category_factors(burn_category)
			  end do
			  burn_check = 1.0
		  end if
     end if


 ! !     check DIFFUSION: to determine whether or not diffusion should happen
 ! !     no diffusion for fully convective, post-MS, and mega-old models
 	  ! s% diffusion_dt_limit = 3.15d7
      !  if(abs(s% mass_conv_core - s% star_mass) < 1d-2) then ! => fully convective
      !     s% diffusion_dt_limit = huge_dt_limit
      !  end if
      !  if (s% star_age > 5d10) then !50 Gyr is really old
      !     s% diffusion_dt_limit = huge_dt_limit
      !  end if
      !  min_center_h1_for_diff = 1d-10
      !  if (s% center_h1 < min_center_h1_for_diff) then
      !     s% diffusion_dt_limit = huge_dt_limit
      !  end if

 !!!! MIST

    !! If abs( extra_energy / star total energy ) > 0.01
    !! reduce tolerance in Tx_emoment root finding
    ! IF ( (ABS(calc_xenergy(id, id_extra)/ s% total_energy) .GT. 0.01) .AND. (s% X_CTRL(7) .GT. 1.E-50) ) THEN
    !     WRITE(*,*) "---*--- Tx tolerance reduced from ", s% X_CTRL(7), "to", 0.1* s% X_CTRL(7)
    !     WRITE(*,*) "---*---     retrying model", s% model_number
    !     s% X_CTRL(7) = 0.1* s% X_CTRL(7)
    !     extras_check_model = retry
    ! END IF

    ! !! If abs( extra_energy / star total energy ) > 0.01
    ! !! decrease mesh in center
    ! IF ( (ABS(calc_xenergy(id, id_extra)/ s% total_energy) .GT. 0.01) ) THEN ! .AND. (X_CTRL(7) .GT. 1.E-10) ) THEN
    !     WRITE(*,*) "---*--- mesh_delta_coeff reduced (for q < 0.2) from ", s% mesh_delta_coeff, "to", 0.75* mesh_delta_coeff
    !     WRITE(*,*) "---*---     retrying model", s% model_number
    !     IF ( (s% q .LT. 0.2) .AND. (s% mesh_delta_coeff .GT. 0.1) ) THEN
    !         s% mesh_delta_coeff = 0.75* s% mesh_delta_coeff
    !     extras_check_model = retry
    ! END IF




      end function extras_check_model


      integer function how_many_extra_history_columns(id, id_extra)
         integer, intent(in) :: id, id_extra
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         how_many_extra_history_columns = 7
      end function how_many_extra_history_columns


      subroutine data_for_extra_history_columns(id, id_extra, n, names, vals, ierr)
         integer, intent(in) :: id, id_extra, n
         character (len=maxlen_history_column_name) :: names(n)
         real(dp) :: vals(n)
         integer, intent(out) :: ierr
         integer :: k
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return

         !note: do NOT add the extras names to history_columns.list
         ! the history_columns.list is only for the built-in log column options.
         ! it must not include the new column names you are adding here.

         names(1) = 'wimp_temp'
         vals(1) = s% X_CTRL(2)
         names(2) = 'Nx_total'
         vals(2) = s% X_CTRL(3)
         names(3) = 'center_nx'
         vals(3) = s% X_CTRL(4)
         names(4) = 'center_np'
         vals(4) = s% X_CTRL(5)
         names(5) = 'Tx_emoment'
         vals(5) = s% X_CTRL(6)
         names(6) = 'extra_energy'
         vals(6) = calc_xenergy(id, id_extra) ! ergs
         names(7) = 'xL/Lnuc'
         vals(7) = s% xtra6

      end subroutine data_for_extra_history_columns


      FUNCTION calc_xenergy(id, id_extra)
          integer, intent(in) :: id, id_extra
          integer :: ierr
          real(dp) :: xe, calc_xenergy
          integer :: k
          type (star_info), pointer :: s
          ierr = 0
          call star_ptr(id, s, ierr)
          if (ierr /= 0) return

          xe = 0.d0
          DO k = 1, s% nz
              xe = xe + s% extra_heat(k)* s% dm(k)* s% dt
          ENDDO

          calc_xenergy = xe ! ergs
      END FUNCTION calc_xenergy



      integer function how_many_extra_profile_columns(id, id_extra)
         use star_def, only: star_info
         integer, intent(in) :: id, id_extra
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         how_many_extra_profile_columns = 3
      end function how_many_extra_profile_columns


      subroutine data_for_extra_profile_columns(id, id_extra, n, nz, names, vals, ierr)
         use star_def, only: star_info, maxlen_profile_column_name
         use const_def, only: dp
         include 'wimp/wimp_vars.h'
         integer, intent(in) :: id, id_extra, n, nz
         integer :: j, idx
         character (len=maxlen_profile_column_name) :: names(n)
         real(dp) :: vals(nz,n)
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         integer :: k
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return

         names(1) = 'nx'
         names(2) = 'np'
         names(3) = 'Vk'
         do k = 1, nz
            vals(k,1) = s% xtra1_array(k)
            vals(k,2) = s% xtra2_array(k)
            vals(k,3) = s% xtra3_array(k)
         end do

      end subroutine data_for_extra_profile_columns


      ! returns either keep_going, retry, backup, or terminate.
      integer function extras_finish_step(id, id_extra)
         use chem_def
!         include 'wimp/wimp_vars.h'
         integer, intent(in) :: id, id_extra
         integer :: ierr, num_dt_low = 0
         LOGICAL :: exist, flg1=.FALSE., flg2=.FALSE., flg3=.FALSE., flg4=.FALSE.
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
!         write('A2' ,'1es16.3') 'Nx',Nx
         if (ierr /= 0) return
         extras_finish_step = keep_going
         call store_extra_info(s)

         s% xtra1 = s% xtra2  !! = Nx (so wimps are not collected when step is not accepted)

!         WRITE(*,*) 'run_star_extras:  Nx s% xtra1 =',s% xtra1, '  Tx s% xtra3 =',s% xtra3

         IF ( (.NOT. flg1) .AND. (s% center_h1 .LT. 0.71D0) ) THEN
         	flg1 = .TRUE.
         	s% need_to_update_history_now = .true.
         	s% need_to_save_profiles_now = .true.
         	s% save_profiles_model_priority = 99	!! ENTER MS
         ENDIF
         IF ( (.NOT. flg2) .AND. (s% center_h1 .LT. 1.D-2) ) THEN
         	flg2 = .TRUE.
         	s% need_to_update_history_now = .true.
         	s% need_to_save_profiles_now = .true.
         	s% save_profiles_model_priority = 98	!! LEAVE MS
         ENDIF
         IF ( (.NOT. flg3) .AND. (s% power_he_burn .GT. 1.D6) ) THEN
         	flg3 = .TRUE.
         	s% need_to_update_history_now = .true.
         	s% need_to_save_profiles_now = .true.
         	s% save_profiles_model_priority = 97	!! He IGNITION
         ENDIF
         IF ( (.NOT. flg4) .AND. (s% center_he4 .LT. 1.D-2) ) THEN
         	flg4 = .TRUE.
         	s% need_to_update_history_now = .true.
         	s% need_to_save_profiles_now = .true.
         	s% save_profiles_model_priority = 96	!! He EXHAUSTED
         ENDIF

         IF ( MOD(s% model_number, 1000) .EQ. 0) THEN
         	s% need_to_update_history_now = .true.
         	s% need_to_save_profiles_now = .true.
         	s% save_profiles_model_priority = 10
         ENDIF

         ! IF ( (ABS(s% xtra6) .GT. 0.085) .AND. (s% xtra4 .GT. 0.2)) THEN
         !     extras_finish_step = terminate
         ! ENDIF
         ! IF ( s% model_number .GT. 1290) THEN
         !     extras_finish_step = terminate
         ! ENDIF


         ! STOPPING CONDITION:
!          IF ((s% star_age .GT. 1.D8) .AND. (s% time_step .LT. 300.D0)) THEN   ! STOPPING CONDITION
!            num_dt_low = num_dt_low+1
!          ELSE
!            num_dt_low = 0
!          ENDIF

!          IF (num_dt_low .GT. 5000) THEN
!              extras_finish_step = terminate
!              s% termination_code = t_xtra1
!              termination_code_str(t_xtra1) = 'dt less than 300 yrs for more than 5000 steps'
!              OPEN(UNIT=10, FILE='README.md', status='old', action='write', position='append')
!              WRITE(10,*) 's% termination_code: ', s% termination_code, &
!                        ' term code str: ', termination_code_str(s% termination_code)
!              CLOSE(UNIT=10)
!              return
!          ENDIF




         ! STOPPING CONDITION:
!          IF ((s% star_age .GT. 1.D8) .AND. (s% time_step .LT. 10.D0)) THEN   ! STOPPING CONDITION
!            num_dt_low = num_dt_low+1
!          ELSE
!            num_dt_low = 0
!          ENDIF

!          IF (num_dt_low .GT. 1.0D5) THEN
!              inquire(file="README.md", exist=exist)
!              IF (exist) then
!                  OPEN(UNIT=10, FILE='README.md', status='old', action='write', position='append')
!              ELSE
!                  OPEN(UNIT=10, FILE='README.md', status='new', action='write')
!              ENDIF
!              extras_finish_step = terminate
!              s% termination_code = t_xtra1
!              termination_code_str(t_xtra1) = 'dt less than 10 yrs for more than 1.0D5 steps'
!              WRITE(10,*) 's% termination_code: ', s% termination_code, &
!                        ' term code str: ', termination_code_str(s% termination_code)
!              CLOSE(UNIT=10)
!              return
!          ENDIF



      end function extras_finish_step



      ! routines for saving and restoring extra data so can do restarts

         ! put these defs at the top and delete from the following routines
         !integer, parameter :: extra_info_alloc = 1
         !integer, parameter :: extra_info_get = 2
         !integer, parameter :: extra_info_put = 3


      subroutine alloc_extra_info(s)
         integer, parameter :: extra_info_alloc = 1
         type (star_info), pointer :: s
         call move_extra_info(s,extra_info_alloc)
      end subroutine alloc_extra_info


      subroutine unpack_extra_info(s)
         integer, parameter :: extra_info_get = 2
         type (star_info), pointer :: s
         call move_extra_info(s,extra_info_get)
      end subroutine unpack_extra_info


      subroutine store_extra_info(s)
         integer, parameter :: extra_info_put = 3
         type (star_info), pointer :: s
         call move_extra_info(s,extra_info_put)
      end subroutine store_extra_info


      subroutine move_extra_info(s,op)
         integer, parameter :: extra_info_alloc = 1
         integer, parameter :: extra_info_get = 2
         integer, parameter :: extra_info_put = 3
         type (star_info), pointer :: s
         integer, intent(in) :: op

         integer :: i, j, num_ints, num_dbls, ierr

         i = 0
         ! call move_int or move_flg
         num_ints = i

         i = 0
         ! call move_dbl

         num_dbls = i

         if (op /= extra_info_alloc) return
         if (num_ints == 0 .and. num_dbls == 0) return

         ierr = 0
         call star_alloc_extras(s% id, num_ints, num_dbls, ierr)
         if (ierr /= 0) then
            write(*,*) 'failed in star_alloc_extras'
            write(*,*) 'alloc_extras num_ints', num_ints
            write(*,*) 'alloc_extras num_dbls', num_dbls
            stop 1
         end if

         contains

         subroutine move_dbl(dbl)
            double precision :: dbl
            i = i+1
            select case (op)
            case (extra_info_get)
               dbl = s% extra_work(i)
            case (extra_info_put)
               s% extra_work(i) = dbl
            end select
         end subroutine move_dbl

         subroutine move_int(int)
            integer :: int
            i = i+1
            select case (op)
            case (extra_info_get)
               int = s% extra_iwork(i)
            case (extra_info_put)
               s% extra_iwork(i) = int
            end select
         end subroutine move_int

         subroutine move_flg(flg)
            logical :: flg
            i = i+1
            select case (op)
            case (extra_info_get)
               flg = (s% extra_iwork(i) /= 0)
            case (extra_info_put)
               if (flg) then
                  s% extra_iwork(i) = 1
               else
                  s% extra_iwork(i) = 0
               end if
            end select
         end subroutine move_flg

      end subroutine move_extra_info

      end module run_star_extras
