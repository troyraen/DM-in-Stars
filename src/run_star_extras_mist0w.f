! ***********************************************************************
!
!   Copyright (C) 2010  Bill Paxton
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

      use star_lib
      use star_def
      use const_def
      use wimp_module   ! necessary to point towards the other_energy hook

      implicit none

      integer :: time0, time1, clock_rate

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
         s% other_wind => Reimers_then_Blocker !MIST

      end subroutine extras_controls


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
         real(dp) :: dt
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         call system_clock(time1,clock_rate)
         dt = dble(time1 - time0) / clock_rate / 60
         write(*,'(/,a50,f12.2,99i10/)') 'runtime (minutes), retries, backups, steps', &
            dt, s% num_retries, s% num_backups, s% model_number
         ierr = 0
      end subroutine extras_after_evolve


      ! returns either keep_going, retry, backup, or terminate.
      integer function extras_check_model(id, id_extra)
         integer, intent(in) :: id, id_extra
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         extras_check_model = keep_going
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
         vals(7) = s% xtra(6)

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
         how_many_extra_profile_columns = 4
      end function how_many_extra_profile_columns


      subroutine data_for_extra_profile_columns(id, id_extra, n, nz, names, vals, ierr)
         use star_def, only: star_info, maxlen_profile_column_name
         use const_def, only: dp
         integer, intent(in) :: id, id_extra, n, nz
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
         names(4) = 'wimp_temp'

         do k = 1, nz
            vals(k,1) = s% xtra1_array(k)
            vals(k,2) = s% xtra2_array(k)
            vals(k,3) = s% xtra3_array(k)
            vals(k,4) = s% X_CTRL(2)
         end do

      end subroutine data_for_extra_profile_columns


      ! returns either keep_going, retry, backup, or terminate.
      integer function extras_finish_step(id, id_extra)
         use chem_def
         integer, intent(in) :: id, id_extra
         integer :: ierr
         LOGICAL :: flg1=.FALSE., flg2=.FALSE., flg3=.FALSE., flg4=.FALSE.
         LOGICAL :: flg5=.FALSE., flg6=.FALSE., flg7=.FALSE., flg8=.FALSE.
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         extras_finish_step = keep_going
         call store_extra_info(s)

         s% xtra(1) = s% xtra(2)  !! = Nx (so wimps are not collected when step is not accepted)

         IF ( (.NOT. flg1) .AND. (s% center_h1 .LT. 0.71D0) ) THEN
           flg1 = .TRUE.
           s% need_to_update_history_now = .true.
           s% need_to_save_profiles_now = .true.
           s% save_profiles_model_priority = 99	!! ENTER MS
         ENDIF
         IF ( (.NOT. flg2) .AND. (s% center_h1 .LT. 0.3D0) ) THEN
           flg2 = .TRUE.
           s% need_to_update_history_now = .true.
           s% need_to_save_profiles_now = .true.
           s% save_profiles_model_priority = 98	!! INTERMEDIATE MS
         ENDIF
         IF ( (.NOT. flg3) .AND. (s% center_h1 .LT. 1.D-1) ) THEN
           flg3 = .TRUE.
           s% need_to_update_history_now = .true.
           s% need_to_save_profiles_now = .true.
           s% save_profiles_model_priority = 97
         ENDIF
         IF ( (.NOT. flg4) .AND. (s% center_h1 .LT. 1.D-2) ) THEN
           flg4 = .TRUE.
           s% need_to_update_history_now = .true.
           s% need_to_save_profiles_now = .true.
           s% save_profiles_model_priority = 96
         ENDIF
         IF ( (.NOT. flg5) .AND. (s% center_h1 .LT. 1.D-3) ) THEN
           flg5 = .TRUE.
           s% need_to_update_history_now = .true.
           s% need_to_save_profiles_now = .true.
           s% save_profiles_model_priority = 95	!! LEAVE MS
         ENDIF
         IF ( (.NOT. flg6) .AND. (s% center_h1 .LT. 1.D-12) ) THEN
           flg6 = .TRUE.
           s% need_to_update_history_now = .true.
           s% need_to_save_profiles_now = .true.
           s% save_profiles_model_priority = 94	!! TAMS
         ENDIF
         IF ( (.NOT. flg7) .AND. (s% power_he_burn .GT. 1.D6) ) THEN
           flg7 = .TRUE.
           s% need_to_update_history_now = .true.
           s% need_to_save_profiles_now = .true.
           s% save_profiles_model_priority = 93	!! He IGNITION
         ENDIF
         IF ( (.NOT. flg8) .AND. (s% center_he4 .LT. 1.D-3) ) THEN
           flg8 = .TRUE.
           s% need_to_update_history_now = .true.
           s% need_to_save_profiles_now = .true.
           s% save_profiles_model_priority = 92	!! He EXHAUSTED
         ENDIF

         IF ( MOD(s% model_number, 1000) .EQ. 0) THEN
           s% need_to_update_history_now = .true.
           s% need_to_save_profiles_now = .true.
           s% save_profiles_model_priority = 10
         ENDIF

         IF ( MOD(s% model_number, 10000) .EQ. 0) THEN
           s% need_to_update_history_now = .true.
           s% need_to_save_profiles_now = .true.
           s% save_profiles_model_priority = 50
         ENDIF

         IF ( MOD(s% model_number, 100000) .EQ. 0) THEN
           s% need_to_update_history_now = .true.
           s% need_to_save_profiles_now = .true.
           s% save_profiles_model_priority = 75
         ENDIF

      end function extras_finish_step


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
            real(dp) :: dbl
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
