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

      use star_lib
      use star_def
      use const_def
      use chem_def
      use wimp_module   ! necessary to point towards the other_energy hook

      implicit none

      integer :: time0, time1, clock_rate
      double precision, parameter :: expected_runtime = 16.5 ! minutes


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
         integer :: ierr, k
         real(dp) :: xengy
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         extras_check_model = keep_going


         ! xlum = 0.0D0 ! check that extra_heat integrates to 0
         ! do k = 1, s% nz
         !     xlum = xlum + (s% extra_heat(k))*(s% dq(k))
         ! end do
         ! IF (xlum.GT.0.2) THEN
         !     write(*,*) '**** Retry because xlum > 0.2. xlum = ', xlum
         !     extras_check_model = retry
         ! ENDIF

      end function extras_check_model


      integer function how_many_extra_history_columns(id, id_extra)
         integer, intent(in) :: id, id_extra
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         how_many_extra_history_columns = 5
      end function how_many_extra_history_columns


      subroutine data_for_extra_history_columns(id, id_extra, n, names, vals, ierr)
         integer, intent(in) :: id, id_extra, n
         character (len=maxlen_history_column_name) :: names(n)
         real(dp) :: vals(n), xe
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
         names(5) = 'extra_energy'
         xe = 0.d0
         DO k = 1, s% nz
             xe = xe + s% extra_heat(k)* s% dm(k)* s% dt
         ENDDO
         vals(5) = xe
!         DO j = 1,10
!            idx = 4+j
!            chemj = s% chem_id(j)
!            names(idx) = chem_isos% name(chemj)
!         ENDDO
!
!         DO idx = 2,15
!             vals(idx-1) = s% x_ctrl(idx)
!         ENDDO

      end subroutine data_for_extra_history_columns

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
