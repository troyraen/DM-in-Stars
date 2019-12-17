! ***********************************************************************
!
!   Copyright (C) 2010  Bill Paxton
!
!   MESA is free software; you can use it and/or modify
!   it under the combined terms and restrictions of the MESA MANIFESTO
!   and the GNU General Library Public License as published
!   by the Free Software Foundation; either version 2 of the License,
!   or (at your option) any later version.
!
!   You should have received a copy of the MESA MANIFESTO along with
!   this software; if not, it is available at the mesa website:
!   http://mesa.sourceforge.net/
!
!   MESA is distributed in the hope that it will be useful,
!   but WITHOUT ANY WARRANTY; without even the implied warranty of
!   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
!   See the GNU Library General Public License for more details.
!
!   You should have received a copy of the GNU Library General Public License
!   along with this software; if not, write to the Free Software
!   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
!
! ***********************************************************************


      module star_lib ! this is the procedure interface to mesa/star.

      use const_def, only: dp
      use star_def, only: star_ptr

      implicit none

      
      contains


      
      
      ! allocate data structures for a star and returns a handle.
      integer function alloc_star(ierr)
         use init, only: alloc_star_data
         integer, intent(out) :: ierr
         alloc_star = alloc_star_data(ierr)
      end function alloc_star

      
      subroutine init_starting_star_data(s, ierr) ! this is done when alloc_star
         ! but if you are reusing the star data (and not calling alloc_star)
         ! then call this to get things initialized.
         use init, only: set_starting_star_data
         use star_def, only: star_info
         type (star_info), pointer :: s
         integer, intent(out) :: ierr
         call set_starting_star_data(s, ierr)
      end subroutine init_starting_star_data
      
      
      ! call this when you are finished with the star.
      subroutine free_star(id, ierr)
         use alloc, only: free_star_data
         ! frees the handle and all associated data
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         
         ierr = 0
         return  ! skip this for now
         
         call free_star_data(id, ierr)
      end subroutine free_star

      
      subroutine read_star_job(s, filename, ierr)
         use star_private_def
         use star_job_ctrls_io, only: do_read_star_job
         type (star_info), pointer :: s
         character(*), intent(in) :: filename
         integer, intent(out) :: ierr
         call do_read_star_job(s, filename, ierr)
      end subroutine read_star_job
      

      subroutine write_star_job(s, filename, ierr)
         use star_private_def
         use star_job_ctrls_io, only: do_write_star_job
         type (star_info), pointer :: s
         character(*), intent(in) :: filename
         integer, intent(out) :: ierr
         call do_write_star_job(s, filename, ierr)
      end subroutine write_star_job


      ! call this after read_star_job.
      ! this sets starlib parameters that apply to all stars.
      ! okay to do extra calls on this; only 1st call is used.
      subroutine starlib_init(s, ierr) 
         use star_def, only: star_info
         type (star_info), pointer :: s
         integer, intent(out) :: ierr
         call do_starlib_init( &
            s% job% mesa_dir, s% job% chem_isotopes_filename, &
            s% job% kappa_file_prefix, s% job% kappa_CO_prefix, s% job% kappa_lowT_prefix, &
            s% job% kappa_blend_logT_upper_bdy, s% job% kappa_blend_logT_lower_bdy, &
            s% job% kappa_type2_logT_lower_bdy, &
            s% job% eos_file_prefix, &
            s% job% eosDT_Z1_suffix, s% job% eosPT_Z1_suffix, &
            s% job% net_reaction_filename, s% job% jina_reaclib_filename, &
            s% job% rate_tables_dir, s% job% rate_cache_suffix, &
            s% job% ionization_file_prefix, s% job% ionization_Z1_suffix, &
            s% job% eosDT_cache_dir, s% job% eosPT_cache_dir, s% job% eosDE_cache_dir, &
            s% job% ionization_cache_dir, s% job% kap_cache_dir, s% job% rates_cache_dir, &
            ierr)
      end subroutine starlib_init


      subroutine do_starlib_init( &
            my_mesa_dir, &
            chem_isotopes_filename, &
            kappa_file_prefix, kappa_CO_prefix, kappa_lowT_prefix, &
            kappa_blend_logT_upper_bdy, kappa_blend_logT_lower_bdy, &
            kappa_type2_logT_lower_bdy, &
            eos_file_prefix, eosDT_Z1_suffix, eosPT_Z1_suffix, &
            net_reaction_filename, jina_reaclib_filename, &
            rate_tables_dir, rates_cache_suffix, &
            ionization_file_prefix, ionization_Z1_suffix, &
            eosDT_cache_dir, eosPT_cache_dir, eosDE_cache_dir, &
            ionization_cache_dir, kap_cache_dir, rates_cache_dir, &
            ierr)
         use init, only: do_star_init
         character (len=*), intent(in) :: &
            my_mesa_dir, chem_isotopes_filename, net_reaction_filename, &
            jina_reaclib_filename, rate_tables_dir, &
            kappa_file_prefix, kappa_CO_prefix, kappa_lowT_prefix, &
            eosDT_Z1_suffix, eosPT_Z1_suffix, &
            eos_file_prefix, rates_cache_suffix, &
            ionization_file_prefix, ionization_Z1_suffix, &
            eosDT_cache_dir, eosPT_cache_dir, eosDE_cache_dir, &
            ionization_cache_dir, kap_cache_dir, rates_cache_dir
         real(dp), intent(in) :: &
            kappa_blend_logT_upper_bdy, kappa_blend_logT_lower_bdy, &
            kappa_type2_logT_lower_bdy
         integer, intent(out) :: ierr
         call do_star_init( &
            my_mesa_dir, &
            chem_isotopes_filename, &
            kappa_file_prefix, kappa_CO_prefix, kappa_lowT_prefix, &
            kappa_blend_logT_upper_bdy, kappa_blend_logT_lower_bdy, &
            kappa_type2_logT_lower_bdy, &
            eos_file_prefix, eosDT_Z1_suffix, eosPT_Z1_suffix, &
            net_reaction_filename, jina_reaclib_filename, &
            rate_tables_dir, rates_cache_suffix, &
            ionization_file_prefix, ionization_Z1_suffix, &
            eosDT_cache_dir, eosPT_cache_dir, eosDE_cache_dir, &
            ionization_cache_dir, kap_cache_dir, rates_cache_dir, &
            ierr)
      end subroutine do_starlib_init
      
      
      ! call this when you are done.
      subroutine starlib_shutdown
         use init, only: do_starlib_shutdown
         call do_starlib_shutdown
      end subroutine starlib_shutdown



      ! if you want direct access to the star data structure, 
      ! then you need to convert the handle to a pointer.
      ! use the routine star_ptr defined in star_def.
      
            
      ! once you've allocated a star, you need to initialize it.
      ! this is done in two stages: first you set the various control parameters
      ! (using star_setup), and then you actually create the model
      ! (using star_load).
      
      
      ! logs and profiles are by default written to the directory named "logs_and_profiles", 
      ! but you can change that if you'd like by calling this routine before calling star_setup.
      subroutine set_dir_for_logs_and_profiles(id, dir_name, ierr)
         use star_def, only: star_info
         integer, intent(in) :: id
         character (len=*), intent(in) :: dir_name
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         s% log_directory = dir_name
      end subroutine set_dir_for_logs_and_profiles
      
      
      subroutine star_set_history_columns(id, history_columns_file, report, ierr)
         use history_specs, only: set_history_columns
         integer, intent(in) :: id
         character (len=*), intent(in) :: history_columns_file
         logical, intent(in) :: report
         integer, intent(out) :: ierr
         call set_history_columns(id, history_columns_file, report, ierr)
      end subroutine star_set_history_columns

      
		integer function star_get_history_column_id(cname)
			! returns id for the history column if there is a matching name
			! returns 0 otherwise.
		   use star_history_def, only: do_get_history_id
			character (len=*), intent(in)  :: cname 
         star_get_history_column_id = do_get_history_id(cname)
      end function star_get_history_column_id
      
      
      subroutine star_set_profile_columns(id, profile_columns_file, report, ierr)
         use profile, only: set_profile_columns
         integer, intent(in) :: id
         character (len=*), intent(in) :: profile_columns_file
         logical, intent(in) :: report
         integer, intent(out) :: ierr
         call set_profile_columns(id, profile_columns_file, report, ierr)
      end subroutine star_set_profile_columns
      
      
      ! read a "namelist" file and setup parameters for the star.
      subroutine star_setup(id, inlist, ierr)
         use ctrls_io, only: do_one_setup
         integer, intent(in) :: id
         character (len=*), intent(in) :: inlist ! can be blank meaning no inlist file
         integer, intent(out) :: ierr ! 0 means AOK.
         call do_one_setup(id, inlist, ierr)
      end subroutine star_setup
      
      
      ! okay to call this more than once; only 1st call does the work.
      subroutine star_set_kap_and_eos_handles(id, ierr)
         use init, only: set_kap_and_eos_handles
         integer, intent(in) :: id
         integer, intent(out) :: ierr ! 0 means AOK.
         call set_kap_and_eos_handles(id, ierr)
      end subroutine star_set_kap_and_eos_handles
      
      
      ! if there is a file called 'restart_photo', then it will be used to restart.
      ! otherwise, create a new model with arbitrary mass and metallicity
      ! as determined by initial_mass and initial_z in the star_info structure.
      ! reads prebuilt initial models from mesa/data/star_data/starting_models.
      ! when star_load returns, the variables in star_def will have been set.
      ! in particular, model_number will be 0 for a fresh start, 
      ! and it will be greater than 0 for a restart.
      subroutine star_load_zams(id, ierr)
         use init, only: load_zams_model
         integer, intent(in) :: id
         integer, intent(out) :: ierr      
         ierr = 0
         call load_zams_model(id, ierr)      
      end subroutine star_load_zams
      
      
      ! you can create a "pre-main-sequence" approximation
      ! that has not started nuclear burning yet.
      ! the following routine will construct a protostar
      ! with uniform composition and center temperature T_c.
      ! the initial_mass and initial_z are specified by the
      ! usual control parameters (e.g., in the inlist file).
      ! T_c must be less than 10^6 so that nuclear burning can be ignored.
      ! rho_c will be adjusted to fit the required mass.
      subroutine star_create_pre_ms_model( &
            id, T_c, guess_rho_c, d_log10_P, logT_surf_limit, &
            logP_surf_limit, pre_ms_initial_zfracs, &
            dump_missing_metals_into_heaviest, &
            change_net, new_net_name, &
            pre_ms_relax_num_steps, ierr)
         use init, only: create_pre_ms_model
         use star_def, only: star_info
         integer, intent(in) :: id
         real(dp), intent(in) :: T_c 
            ! optional initial center temperature
            ! set to 0 to use default
         real(dp), intent(in) :: guess_rho_c 
            ! optional initial guess for center density
            ! set to 0 to use default
         real(dp), intent(in) :: d_log10_P 
            ! standard point spacing in initial model is d_log10_P
            ! set to 0 to use default
         ! model contruction is from inside out and stops when at either of the following.
         real(dp), intent(in) :: logT_surf_limit 
            ! set to 0 to use default
         real(dp), intent(in) :: logP_surf_limit 
            ! set to 0 to use default
         integer, intent(in) :: pre_ms_initial_zfracs, pre_ms_relax_num_steps 
         logical, intent(in) :: dump_missing_metals_into_heaviest, change_net
         character(len=*), intent(in) :: new_net_name
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         s% pre_ms_T_c = T_c
         s% pre_ms_guess_rho_c = guess_rho_c
         s% pre_ms_d_log10_P = d_log10_P
         s% pre_ms_logT_surf_limit = logT_surf_limit
         s% pre_ms_logP_surf_limit = logP_surf_limit
         s% pre_ms_initial_zfracs = pre_ms_initial_zfracs
         s% pre_ms_change_net = change_net
         s% pre_ms_new_net_name = new_net_name
         s% pre_ms_relax_num_steps = pre_ms_relax_num_steps
         s% pre_ms_dump_missing_heaviest = dump_missing_metals_into_heaviest
         call create_pre_ms_model(id, ierr)
         if (ierr /= 0) return
      end subroutine star_create_pre_ms_model
      
      
      ! you can create an initial model for given mass and radius.
      subroutine star_create_initial_model(id, &
            radius_in_cm_for_create_initial_model, &
            mass_in_gm_for_create_initial_model, &
            center_logP_1st_try_for_create_initial_model, &
            entropy_1st_try_for_create_initial_model, &
            max_tries_for_create_initial_model, &
            abs_e01_tolerance_for_create_initial_model, &
            abs_e02_tolerance_for_create_initial_model, &            
            initial_zfracs, dump_missing_metals_into_heaviest, change_net, new_net_name, &
            initial_model_relax_num_steps, initial_model_eps, ierr)
         use init, only: create_initial_model
         use star_def, only: star_info
         integer, intent(in) :: id
         real(dp), intent(in) :: radius_in_cm_for_create_initial_model, &
            mass_in_gm_for_create_initial_model, &
            center_logP_1st_try_for_create_initial_model, &
            entropy_1st_try_for_create_initial_model, &
            abs_e01_tolerance_for_create_initial_model, &
            abs_e02_tolerance_for_create_initial_model
         integer, intent(in) :: &
            initial_zfracs, initial_model_relax_num_steps, max_tries_for_create_initial_model
         logical, intent(in) :: dump_missing_metals_into_heaviest, change_net
         character(len=*), intent(in) :: new_net_name
         real(dp), intent(in) :: initial_model_eps 
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         s% radius_in_cm_for_create_initial_model = radius_in_cm_for_create_initial_model
         s% mass_in_gm_for_create_initial_model = mass_in_gm_for_create_initial_model
         s% center_logP_1st_try_for_create_initial_model = &
            center_logP_1st_try_for_create_initial_model
         s% entropy_1st_try_for_create_initial_model = &
            entropy_1st_try_for_create_initial_model
         s% max_tries_for_create_initial_model = max_tries_for_create_initial_model
         s% abs_e01_tolerance_for_create_initial_model = &
            abs_e01_tolerance_for_create_initial_model
         s% abs_e02_tolerance_for_create_initial_model = &
            abs_e02_tolerance_for_create_initial_model         
         s% initial_zfracs_for_create_initial_model = initial_zfracs
         s% initial_model_relax_num_steps = initial_model_relax_num_steps
         s% initial_model_eps = initial_model_eps
         s% initial_model_change_net = change_net
         s% initial_model_new_net_name = new_net_name
         s% initial_dump_missing_heaviest = dump_missing_metals_into_heaviest
         call create_initial_model(id, ierr)
         if (ierr /= 0) return
      end subroutine star_create_initial_model
      
      
      subroutine star_allocate_arrays(s, ierr)
         ! for use by other_build_initial_model
         use alloc, only: allocate_star_info_arrays
         use star_def, only: star_info
         type (star_info), pointer :: s
         integer, intent(out) :: ierr
         call allocate_star_info_arrays(s, ierr)
      end subroutine star_allocate_arrays
      

      logical function doing_a_restart(restart_filename)
         use init, only: doing_restart
         character (len=*) :: restart_filename
         doing_a_restart = doing_restart(restart_filename)
      end function doing_a_restart


      subroutine star_load_restart_photo(id, restart_filename, ierr)
         use init, only: load_restart_photo
         integer, intent(in) :: id
         character (len=*), intent(in) :: restart_filename
         integer, intent(out) :: ierr      
         call load_restart_photo(id, restart_filename, ierr)      
      end subroutine star_load_restart_photo
      
      
      subroutine star_write_model(id, filename, ierr)
         use write_model, only: do_write_model
         integer, intent(in) :: id
         character (len=*), intent(in) :: filename
         integer, intent(out) :: ierr
         call do_write_model(id, filename, ierr)      
      end subroutine star_write_model
      
      
      subroutine star_read_model(id, model_fname, ierr)
         use init, only: load_saved_model
         integer, intent(in) :: id
         character (len=*), intent(in) :: model_fname
         integer, intent(out) :: ierr      
         call load_saved_model(id, model_fname, ierr)     
      end subroutine star_read_model
      
      
      subroutine star_number_from_saved_model(fname, model_number, ierr)
         use read_model, only: do_read_saved_model_number
         character (len=*), intent(in) :: fname ! filename for the saved model
         integer, intent(inout) :: model_number 
            ! set only if this property is present in file
         integer, intent(out) :: ierr
         call do_read_saved_model_number(fname, model_number, ierr)
      end subroutine star_number_from_saved_model
      
      
      subroutine star_age_from_saved_model(fname, star_age, ierr)
         use read_model, only: do_read_saved_model_age
         character (len=*), intent(in) :: fname ! filename for the saved model
         real(dp), intent(inout) :: star_age 
            ! set only if this property is present in file
         integer, intent(out) :: ierr
         call do_read_saved_model_age(fname, star_age, ierr)
      end subroutine star_age_from_saved_model
      
      
            
      ! after you've created a starting model, you're ready to evolve it.
      ! this process is done one step at a time by calling star_evolve_step.
      
      
      ! this routine takes one step in the evolution.
      ! when it returns successfully (i.e, with value = keep_going), the data
      ! describing the new model can be found in the variables defined in star_def.
      integer function star_evolve_step(id, first_try, just_did_backup)
         ! returns either keep_going, redo, retry, backup, or terminate
         use star_def, only: terminate, keep_going
         use star_def, only: star_info
         use evolve, only: do_evolve_step_part1, do_evolve_step_part2
         use star_utils, only: start_time, update_time
         integer, intent(in) :: id
         logical, intent(in) :: first_try 
            ! true on the first try to take this step
            ! false if this is a repeat for a retry or a backup
         logical, intent(in) :: just_did_backup
         type (star_info), pointer :: s
         integer :: ierr, time0, clock_rate
         real(dp) :: total
         star_evolve_step = terminate
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         if (s% doing_timing) call start_time(s, time0, total)
         star_evolve_step = do_evolve_step_part1(id, first_try, just_did_backup)
         if (star_evolve_step == keep_going) &
            star_evolve_step = do_evolve_step_part2(id, first_try, just_did_backup)
         if (s% doing_timing) call update_time(s, time0, total, s% time_evolve_step)         
      end function star_evolve_step

      ! individual functions to evolve each of the parts of star_evolve_step
      integer function star_evolve_step_part1(id, first_try, just_did_backup)
         use star_def, only: terminate
         use star_def, only: star_info
         use evolve, only: do_evolve_step_part1
         integer, intent(in) :: id
         logical, intent(in) :: first_try 
         logical, intent(in) :: just_did_backup
         type (star_info), pointer :: s
         integer :: ierr
         star_evolve_step_part1 = terminate
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         star_evolve_step_part1 = do_evolve_step_part1(id, first_try, just_did_backup)
      end function star_evolve_step_part1

      integer function star_evolve_step_part2(id, first_try, just_did_backup)
         use star_def, only: terminate
         use star_def, only: star_info
         use evolve, only: do_evolve_step_part2
         integer, intent(in) :: id
         logical, intent(in) :: first_try 
         logical, intent(in) :: just_did_backup
         type (star_info), pointer :: s
         integer :: ierr
         star_evolve_step_part2 = terminate
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         star_evolve_step_part2 = do_evolve_step_part2(id, first_try, just_did_backup)
      end function star_evolve_step_part2
      
      ! once the step is completed, call the following routines to check the
      ! new model and pick the next timestep.
      
      
      ! this is the standard routine for checking the model after each step.
      ! it takes care of things such as writing logs and profiles.
      integer function star_check_model(id)
         ! returns either keep_going, redo, retry, backup, or terminate.
         use do_one_utils, only: do_one_check_model
         integer, intent(in) :: id
         star_check_model = do_one_check_model(id)
      end function star_check_model
      
      
      ! this is the standard routine for checking if have reached some limit
      ! such as max_age, max_model_number, psi_center_limit, h1_center_limit, etc.
      integer function star_check_limits(id)
         ! returns either keep_going or terminate.
         use do_one_utils, only: do_check_limits
         integer, intent(in) :: id
         star_check_limits = do_check_limits(id)
      end function star_check_limits
      
      
      ! this routine inspects the new model and picks a new timestep.
      ! if it decides that the changes in the new model are too great, 
      ! it will return retry or backup.
      integer function star_pick_next_timestep(id)
         ! returns either keep_going, redo, retry, backup, or terminate.
         use evolve, only: pick_next_timestep
         integer, intent(in) :: id
         star_pick_next_timestep = pick_next_timestep(id)
      end function star_pick_next_timestep
      
      
      ! at the end of a successful step, call this routine to take care of
      ! things such as writing log files or saving a "snapshot" for restarts.
      integer function star_finish_step( &
            id, id_extra, do_photo, &
            how_many_extra_profile_columns, data_for_extra_profile_columns, &
            how_many_extra_history_columns, data_for_extra_history_columns, &
            ierr)
         ! returns either keep_going, redo, retry, backup, or terminate.
         use evolve, only: finish_step
         integer, intent(in) :: id, id_extra
         logical, intent(in) :: do_photo ! if true, then save "photo" for restart
         interface
            include 'extra_profile_cols.inc'
            include 'extra_history_cols.inc'
         end interface
         integer, intent(out) :: ierr
         star_finish_step = &
            finish_step(id, id_extra, do_photo, &
               how_many_extra_profile_columns, data_for_extra_profile_columns, &
               how_many_extra_history_columns, data_for_extra_history_columns, ierr)
      end function star_finish_step

      
      
      ! this routine needs to be called before you do a redo.
      integer function star_prepare_to_redo(id)
         ! returns either keep_going, retry, backup, or terminate.
         use evolve, only: prepare_to_redo
         integer, intent(in) :: id
         star_prepare_to_redo = prepare_to_redo(id)
      end function star_prepare_to_redo

      
      
      ! once in a while an attempted step will fail, and you'll need to retry it 
      ! with a smaller timestep or resort to backing up to a previous model.
      
      
      ! this routine needs to be called before you do a retry.
      integer function star_prepare_to_retry(id)
         ! returns either keep_going, backup, or terminate.
         use evolve, only: prepare_to_retry
         integer, intent(in) :: id
         star_prepare_to_retry = prepare_to_retry(id)
      end function star_prepare_to_retry
      
      
      ! this routine needs to be called before you do a backup.
      integer function star_do1_backup(id)
         ! returns either keep_going or terminate.
         use evolve, only: do1_backup
         integer, intent(in) :: id
         star_do1_backup = do1_backup(id)
      end function star_do1_backup
      
      
      ! at the end of the evolution run, call this to do things such as
      ! closing log files.
      subroutine star_after_evolve(id, ierr)
         use do_one_utils, only: do_one_finish
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         call do_one_finish(id, ierr)
      end subroutine star_after_evolve
      


      ! typically, after the namelist controls file has been read by the star setup routine, 
      ! you won't need to do anything else with it.   But in case you want
      ! to read or write a control file at other times, here are the routines to do it.      
      subroutine star_read_controls(id, filename, ierr)
         use ctrls_io, only: read_controls
         integer, intent(in) :: id
         character(*), intent(in) :: filename
         integer, intent(out) :: ierr
         call read_controls(id, filename, ierr)
      end subroutine star_read_controls
      
      
      ! NOTE: you can create an atmosphere for use with pulsation codes.
      ! inspired by B. Paczynski, 1969, Acta Astr., vol. 19, 1.
      ! takes into account dilution of luminosity when tau < 2/3,
      ! and calls mlt to get gradT allowing for convection.
      ! you should use the Henyey option for mlt when adding an atmosphere.
      
      subroutine star_write_fgong(id, &
            add_center_point, keep_surface_point, &
            add_atmosphere, &
            filename, ierr)
         ! writes the model in FGONG format for use with pulsation codes
         use pulsation_info, only: save_pulsation_info
         integer, intent(in) :: id
         logical, intent(in) :: &
            add_center_point, keep_surface_point, &
            add_atmosphere
         character(*), intent(in) :: filename
         integer, intent(out) :: ierr
         call save_pulsation_info( &
            id, 'fgong', &
            add_center_point, keep_surface_point, &
            add_atmosphere, &
            filename, ierr)
      end subroutine star_write_fgong
      
      
      subroutine star_write_osc(id, &
            add_center_point, keep_surface_point, &
            add_atmosphere, filename, ierr)
         ! writes the model in OSC format for use with pulsation codes
         use pulsation_info, only: save_pulsation_info
         integer, intent(in) :: id
         logical, intent(in) :: &
            add_center_point, keep_surface_point, &
            add_atmosphere
         character(*), intent(in) :: filename
         integer, intent(out) :: ierr
         call save_pulsation_info( &
            id, 'osc', &
            add_center_point, keep_surface_point, &
            add_atmosphere, filename, ierr)
      end subroutine star_write_osc
      
      
      subroutine star_write_gyre(id, &
            keep_surface_point, add_atmosphere, filename, ierr)
         ! writes the model in GYRE format for use with pulsation codes
         use pulsation_info, only: save_pulsation_info
         integer, intent(in) :: id
         logical, intent(in) :: keep_surface_point, add_atmosphere
         character(*), intent(in) :: filename
         integer, intent(out) :: ierr
         logical, parameter :: add_center_point = .false.
         call save_pulsation_info( &
            id, 'gyre', &
            add_center_point, keep_surface_point, &
            add_atmosphere, filename, ierr)
      end subroutine star_write_gyre
      
      
      subroutine star_write_gyre_data_to_file(filename,  &
            M_star, R_star, L_star, &
            r, w, L, p, rho, T, &
            N2, Gamma_1, nabla_ad, delta, nabla,  &
            kappa, kappa_rho, kappa_T, &
            epsilon, epsilon_rho, epsilon_T, omega, &
            max_num_gyre_points, ierr)
         use pulsation_info, only: do_write_gyre
         character (len=*), intent(in) :: filename
         real(dp), intent(in) :: M_star, R_star, L_star
         real(dp), dimension(:), pointer, intent(in) :: &
            r, w, L, p, rho, T, &
            N2, Gamma_1, nabla_ad, delta, nabla,  &
            kappa, kappa_rho, kappa_T, &
            epsilon, epsilon_rho, epsilon_T, omega
         integer, intent(in) :: max_num_gyre_points
         integer, intent(out) :: ierr
         call do_write_gyre(filename,  &
            M_star, R_star, L_star, &
            r, w, L, p, rho, T, &
            N2, Gamma_1, nabla_ad, delta, nabla,  &
            kappa, kappa_rho, kappa_T, &
            epsilon, epsilon_rho, epsilon_T, omega, &
            max_num_gyre_points, ierr)
      end subroutine star_write_gyre_data_to_file


      subroutine star_write_pulsation_info(id, &
            add_center_point, keep_surface_point, &
            add_atmosphere, &
            pulsation_info_format, filename, ierr)
         ! writes the model in specified format for use with pulsation codes
         use pulsation_info, only: save_pulsation_info
         integer, intent(in) :: id
         logical, intent(in) :: &
            add_center_point, keep_surface_point, &
            add_atmosphere
         character(*), intent(in) :: pulsation_info_format, filename
         integer, intent(out) :: ierr
         call save_pulsation_info( &
            id, pulsation_info_format, &
            add_center_point, keep_surface_point, &
            add_atmosphere, &
            filename, ierr)
      end subroutine star_write_pulsation_info
      
      
      subroutine star_get_fgong_info(id, &
            add_center_point, keep_surface_point, &
            add_atmosphere, &
            nn, iconst, ivar, glob, var, ierr)
         ! return the model in FGONG format
         use pulsation_info, only: get_fgong_info
         integer, intent(in) :: id
         logical, intent(in) :: &
            add_center_point, keep_surface_point, &
            add_atmosphere
         integer, intent(out) :: nn, iconst, ivar
         real(dp), pointer :: glob(:) ! (iconst) -- will be allocated
         real(dp), pointer :: var(:,:) ! (ivar,nn) -- will be allocated
         integer, intent(out) :: ierr
         call get_fgong_info(id, &
            add_center_point, keep_surface_point, &
            add_atmosphere, &
            nn, iconst, ivar, glob, var, ierr)
      end subroutine star_get_fgong_info
      
      
      subroutine star_get_osc_info(id, &
            add_center_point, keep_surface_point, add_atmosphere, &
            nn, iconst, ivar, glob, var, ierr)
         ! return the model in OSC format
         use pulsation_info, only: get_osc_info
         integer, intent(in) :: id
         logical, intent(in) :: add_center_point, keep_surface_point, add_atmosphere
         integer, intent(out) :: nn, iconst, ivar
         real(dp), pointer :: glob(:) ! (iconst) -- will be allocated
         real(dp), pointer :: var(:,:) ! (ivar,nn) -- will be allocated
         integer, intent(out) :: ierr
         call get_osc_info(id, &
            add_center_point, keep_surface_point, add_atmosphere, &
            nn, iconst, ivar, glob, var, ierr)
      end subroutine star_get_osc_info
      
      
      subroutine star_get_gyre_info( &
            id, keep_surface_point, add_atmosphere, &
            M_star, R_star, L_star, r, w, L, p, rho, T, &
            N2, Gamma_1, nabla_ad, delta, nabla,  &
            kappa, kappa_rho, kappa_T, &
            epsilon, epsilon_rho, epsilon_T, omega, ierr)
         ! return the model in GYRE format
         use pulsation_info, only: get_gyre_model
         integer, intent(in) :: id
         logical, intent(in) :: keep_surface_point, add_atmosphere
         real(dp), intent(out) :: M_star, R_star, L_star
         ! arrays will be allocated; caller must deallocate when done with them
         real(dp), dimension(:), pointer :: &
            r, w, L, p, rho, T, &
            N2, Gamma_1, nabla_ad, delta, nabla,  &
            kappa, kappa_rho, kappa_T, &
            epsilon, epsilon_rho, epsilon_T, omega
         integer, intent(out) :: ierr
         call get_gyre_model( &
            id, keep_surface_point, add_atmosphere, &
            M_star, R_star, L_star, r, w, L, p, rho, T, &
            N2, Gamma_1, nabla_ad, delta, nabla,  &
            kappa, kappa_rho, kappa_T, &
            epsilon, epsilon_rho, epsilon_T, omega, ierr)
      end subroutine star_get_gyre_info
            
      
      subroutine star_get_pulsation_info( &
            id, pulse_info_format, &
            add_center_point, keep_surface_point, &
            add_atmosphere, &
            nn, iconst, ivar, glob, var, ierr)
         use pulsation_info, only: get_pulsation_format, get_pulsation_info
         integer, intent(in) :: id
         character (len=*), intent(in) :: pulse_info_format
         logical, intent(in) :: &
            add_center_point, keep_surface_point, &
            add_atmosphere
         integer, intent(out) :: nn, iconst, ivar
         real(dp), pointer :: glob(:) ! (iconst) -- will be allocated
         real(dp), pointer :: var(:,:) ! (ivar,nn) -- will be allocated
         integer, intent(out) :: ierr
         integer :: which_format
         which_format = get_pulsation_format(pulse_info_format, ierr)
         if (ierr /= 0) return
         call get_pulsation_info(id, which_format, &
            add_center_point, keep_surface_point, &
            add_atmosphere, &
            nn, iconst, ivar, glob, var, ierr)
      end subroutine star_get_pulsation_info
      
      
      subroutine star_write_short_format(id, filename, ierr)
         ! writes the model in .short format for use with GR1D
         use pulsation_info, only: do_write_short_format
         integer, intent(in) :: id
         character(*), intent(in) :: filename
         integer, intent(out) :: ierr
         call do_write_short_format(id, filename, ierr)
      end subroutine star_write_short_format
            
      
      subroutine star_create_atm(s, ierr)
         ! sets s% atm_structure_num_pts and s% atm_structure
         ! if which_atm_option is an integration, then uses that.
         ! else uses Paczynski.
         use star_def, only: star_info
         use create_atm, only: do_create_atm
         type (star_info), pointer :: s
         integer, intent(out) :: ierr
         call do_create_atm(s, ierr)
      end subroutine star_create_atm

      
      ! normally, "snapshots" for restarts will be saved automatically according
      ! to the value of the photostep parameter.  but if you want to 
      ! do it yourself, you can call the following routine.      
      subroutine star_save_for_restart(id, filename, ierr)
         use evolve_support, only: output_to_file
         integer, intent(in) :: id
         character (len=*) :: filename
         integer, intent(out) :: ierr
         call output_to_file(filename, id, ierr)
      end subroutine star_save_for_restart
      
      
      integer function num_standard_history_columns(s) ! not inluding any extra columns
         use star_def, only: star_info
         type (star_info), pointer :: s
         num_standard_history_columns = size(s% history_column_spec, dim=1)
      end function num_standard_history_columns
      
      
      ! set "history info" in star data
      subroutine get_data_for_history_columns(s, id_extra, &
            how_many_extra_history_columns, &
            data_for_extra_history_columns, &
            ierr)
         use history, only: do_get_data_for_history_columns
         use star_def, only: star_info, maxlen_profile_column_name
         type (star_info), pointer :: s
         integer, intent(in) :: id_extra
         interface
            include 'extra_history_cols.inc'
         end interface
         integer, intent(out) :: ierr
         call do_get_data_for_history_columns( &
            s, id_extra, &
            how_many_extra_history_columns, data_for_extra_history_columns, &
            ierr)
      end subroutine get_data_for_history_columns
      
      
      integer function num_standard_profile_columns(s) ! not inluding any extra profile columns
         use star_def, only: star_info
         type (star_info), pointer :: s
         num_standard_profile_columns = size(s% profile_column_spec, dim=1)
      end function num_standard_profile_columns
      
      
      subroutine get_data_for_profile_columns(s, id_extra, &
            how_many_extra_profile_columns, data_for_extra_profile_columns, &
            numvals, nz, names, vals, is_int, ierr)
         use profile, only: do_get_data_for_profile_columns
         use star_def, only: star_info, maxlen_profile_column_name
         type (star_info), pointer :: s
         integer, intent(in) :: id_extra, numvals, nz
         interface
            include 'extra_profile_cols.inc'
         end interface
         ! note: it is the caller's job to allocate names and vals before calling
         ! and deallocate them when done.
         character (len=maxlen_profile_column_name), pointer :: names(:) ! (num_columns)
         real(dp), pointer :: vals(:,:) ! (nz,num_columns)
         logical, pointer :: is_int(:) ! (num_columns) true iff the values in the column are integers
         integer, intent(out) :: ierr
         call do_get_data_for_profile_columns(s, id_extra, numvals, nz, &
            how_many_extra_profile_columns, data_for_extra_profile_columns, &
            names, vals, is_int, ierr)
      end subroutine get_data_for_profile_columns
      
      
      
      ! you may want to have some data automatically saved and restored along with
      ! the rest of the information in a snapshot.  you can do it by using the following routines.
      ! for example, you can check the model_number after star_load returns to see if you
      ! are doing a fresh start or a restart.  If the model_number is 0, it is a fresh start and
      ! you can call star_alloc_extras to create the arrays and then call star_extra_arrays to
      ! get pointers to them.  The contents of the arrays will be saved as part of any future snapshot.
      ! If the model_number is greater than 0 when star_load returns, then skip the call on
      ! star_alloc_extras because the arrays will have been automatically allocated and restored as part of
      ! the restart process.  Call star_extra_arrays to get pointers to the arrays which will
      ! have the same content as when the snapshot was made.
      ! the routine star_finish_step will save the contents of the extra arrays along with
      ! the rest of the information for a restart.
      ! the routine star_load will restore the contents of the arrays when there is a restart.
      ! see star/test/src/rlo_exp.f for an example that uses this scheme.
      subroutine star_alloc_extras(id, len_extra_iwork, len_extra_work, ierr)
         use alloc, only: alloc_extras
         integer, intent(in) :: id
         integer, intent(in) :: len_extra_iwork, len_extra_work
         integer, intent(out) :: ierr
         call alloc_extras(id, len_extra_iwork, len_extra_work, ierr)
      end subroutine star_alloc_extras
      
      
      ! if for some reason, you're no longer interested in having extra arrays, you can call this.
      ! it is called automatically when you call free_star, so for normal use, you don't need to
      ! worry about deallocating extra arrays when you are finished with a star.
      subroutine star_dealloc_extras(id)
         use star_def, only: star_info
         use alloc, only: dealloc_extras
         integer, intent(in) :: id
         type (star_info), pointer :: s
         integer :: ierr
         call star_ptr(id,s,ierr)
         if (ierr /= 0) return
         call dealloc_extras(s)
      end subroutine star_dealloc_extras
      
      
      subroutine star_set_age(id, age, ierr)
         use evolve, only: set_age
         integer, intent(in) :: id
         real(dp), intent(in) :: age ! in years
         integer, intent(out) :: ierr
         call set_age(id, age, ierr)
      end subroutine star_set_age
      
      
      ! this routine is for changing use of lnPgas variables.
      ! simply changes variables; doesn't reconverge the model.
      subroutine star_set_lnPgas_flag(id, lnPgas_flag, ierr)
         use alloc, only: set_lnPgas_flag
         integer, intent(in) :: id
         logical, intent(in) :: lnPgas_flag
         integer, intent(out) :: ierr
         call set_lnPgas_flag(id, lnPgas_flag, ierr)
      end subroutine star_set_lnPgas_flag
      
      
      ! this routine is for changing use of internal energy variables.
      ! simply changes variables; doesn't reconverge the model.
      subroutine star_set_E_flag(id, E_flag, ierr)
         use alloc, only: set_E_flag
         integer, intent(in) :: id
         logical, intent(in) :: E_flag
         integer, intent(out) :: ierr
         call set_E_flag(id, E_flag, ierr)
      end subroutine star_set_E_flag
      
      
      ! this routine is for adding or removing luminosity variables.
      ! simply adds or removes; doesn't reconverge the model.
      subroutine star_set_L_flag(id, L_flag, ierr)
         use alloc, only: set_L_flag
         integer, intent(in) :: id
         logical, intent(in) :: L_flag
         integer, intent(out) :: ierr
         call set_L_flag(id, L_flag, ierr)
      end subroutine star_set_L_flag
      
      
      ! this routine is for adding or removing velocity variables.
      ! simply adds or removes; doesn't reconverge the model.
      subroutine star_set_v_flag(id, v_flag, ierr)
         use alloc, only: set_v_flag
         integer, intent(in) :: id
         logical, intent(in) :: v_flag
         integer, intent(out) :: ierr
         call set_v_flag(id, v_flag, ierr)
      end subroutine star_set_v_flag
      
      
      ! this routine is for adding or removing rotation variables.
      ! simply adds or removes; doesn't reconverge the model.
      subroutine star_set_rotation_flag(id, rotation_flag, ierr)
         use alloc, only: set_rotation_flag
         integer, intent(in) :: id
         logical, intent(in) :: rotation_flag
         integer, intent(out) :: ierr
         call set_rotation_flag(id, rotation_flag, ierr)
      end subroutine star_set_rotation_flag
      
      
      ! you can change the nuclear net at the start or between steps
      ! added species are given initial abundances based on solar scaled by initial_z
      
      subroutine star_change_to_new_net( &
            id, adjust_abundances_for_new_isos, new_net_name, ierr)
         use adjust_xyz, only: change_net
         integer, intent(in) :: id
         logical, intent(in) :: adjust_abundances_for_new_isos
         character (len=*), intent(in) :: new_net_name
         integer, intent(out) :: ierr
         call change_net( &
            id, adjust_abundances_for_new_isos, new_net_name, ierr)
      end subroutine star_change_to_new_net
      
      
      ! Heger-style adaptive network (Woosley, Heger, et al, ApJSS, 151:75-102, 2004)
      subroutine star_adjust_net(id, &
            min_x_for_keep, min_x_for_n, min_x_for_add, max_Z, max_N, max_A, ierr)
         use star_def, only: star_info
         use adjust_net, only: check_adjust_net
         integer, intent(in) :: id
         real(dp), intent(in) :: &
            min_x_for_keep, min_x_for_n, min_x_for_add, max_Z, max_N, max_A
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         call check_adjust_net(s, s% species, &
            min_x_for_keep, min_x_for_n, min_x_for_add, &
            max_Z, max_N, max_A, ierr)
      end subroutine star_adjust_net
            
      
      logical function is_included_in_net(id, species, ierr)
         use star_def, only: star_info
         integer, intent(in) :: id
         integer, intent(in) :: species ! a chem_id such as ihe3.  see chem_def.
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         call star_ptr(id, s, ierr)
         if (ierr /= 0) then
            is_included_in_net = .false.
            return
         end if
         is_included_in_net = (s% net_iso(species) /= 0)
      end function is_included_in_net
            
      
      ! here are some routines for doing special adjustments to the star's composition

      
      ! set uniform composition with one of the standard metal z fractions from chem_def
      subroutine star_set_standard_composition(id, h1, h2, he3, he4, &
            which_zfracs, dump_missing_metals_into_heaviest, ierr)
         use adjust_xyz, only: set_standard_composition
         use star_def, only: star_info
         integer, intent(in) :: id
         real(dp), intent(in) :: h1, h2, he3, he4 ! mass fractions
         integer, intent(in) :: which_zfracs ! defined in chem_def. e.g., GS98_zfracs
         logical, intent(in) :: dump_missing_metals_into_heaviest
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         call set_standard_composition(s, s% species, h1, h2, he3, he4, &
            which_zfracs, dump_missing_metals_into_heaviest, ierr)
      end subroutine star_set_standard_composition
      

      subroutine star_uniform_xa_from_file(id, file_for_uniform_xa, ierr)
         use adjust_xyz, only: set_uniform_xa_from_file
         integer, intent(in) :: id
         character (len=*), intent(in) :: file_for_uniform_xa
         integer, intent(out) :: ierr
         call set_uniform_xa_from_file(id, file_for_uniform_xa, ierr)
      end subroutine star_uniform_xa_from_file


      subroutine star_set_uniform_composition(id, species, xa, ierr)
         use adjust_xyz, only: set_uniform_composition
         integer, intent(in) :: id
         integer, intent(in) :: species
         real(dp), intent(in) :: xa(species)
         integer, intent(out) :: ierr
         call set_uniform_composition(id, species, xa, ierr)
      end subroutine star_set_uniform_composition
      
      
      subroutine star_set_composition(id, species, xa, ierr)
         use adjust_xyz, only: set_composition
         use star_def, only: star_info
         integer, intent(in) :: id
         integer, intent(in) :: species
         real(dp), intent(in) :: xa(species) ! the replacement mass fractions
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         call set_composition(id, 1, s% nz, species, xa, ierr)
      end subroutine star_set_composition
      
      
      subroutine set_composition_in_section(id, nzlo, nzhi, species, xa, ierr)
         use adjust_xyz, only: set_composition
         integer, intent(in) :: id
         integer, intent(in) :: nzlo, nzhi ! change cells from nzlo to nzhi, inclusive.
         integer, intent(in) :: species
         real(dp), intent(in) :: xa(species) ! cells from nzlo to nzhi get this composition.
         integer, intent(out) :: ierr
         call set_composition(id, nzlo, nzhi, species, xa, ierr)
      end subroutine set_composition_in_section
      
      
      subroutine change_to_xa_for_accretion(id, nzlo, nzhi, ierr)
         use adjust_xyz, only: do_change_to_xa_for_accretion
         integer, intent(in) :: id
         integer, intent(in) :: nzlo, nzhi ! change cells from nzlo to nzhi, inclusive.
         integer, intent(out) :: ierr
         call do_change_to_xa_for_accretion(id, nzlo, nzhi, ierr)
      end subroutine change_to_xa_for_accretion
      
      
      subroutine star_set_abundance_ratio(id, i1, i2, ratio, ierr)
         use adjust_xyz, only: set_abundance_ratio
         use star_def, only: star_info
         integer, intent(in) :: id
         integer, intent(in) :: i1, i2 ! chem id's such as ih1 or ihe4 from chem_def
         real(dp), intent(in) :: ratio ! change abundances of i1 and i2 s.t. x(i1)/x(i2)=ratio
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         call set_abundance_ratio(id, i1, i2, ratio, 1, s% nz, ierr)
      end subroutine star_set_abundance_ratio
      
      
      subroutine set_abundance_ratio_in_section(id, i1, i2, ratio, nzlo, nzhi, ierr)
         use adjust_xyz, only: set_abundance_ratio
         integer, intent(in) :: id
         integer, intent(in) :: i1, i2 ! chem id's such as ih1 or ihe4 from chem_def
         real(dp), intent(in) :: ratio ! change abundances of i1 and i2 s.t. x(i1)/x(i2)=ratio
         integer, intent(in) :: nzlo, nzhi ! change cells from nzlo to nzhi, inclusive.
         integer, intent(out) :: ierr
         call set_abundance_ratio(id, i1, i2, ratio, nzlo, nzhi, ierr)
      end subroutine set_abundance_ratio_in_section
      
      
      subroutine star_set_y(id, y, ierr)
         ! changes abundances of h1 and he4 only
         ! adjust ratio of h1 to he4 to be (1-y-z)/y at each point
         use adjust_xyz, only: set_y
         use star_def, only: star_info
         integer, intent(in) :: id
         real(dp), intent(in) :: y ! new value for average he4 mass fraction
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         call set_y(s, y, 1, s% nz, ierr)
      end subroutine star_set_y
      
      
      subroutine set_y_in_section(id, y, nzlo, nzhi, ierr)
         ! change abundances of h1 and he4
         use adjust_xyz, only: set_y
         use star_def, only: star_info
         integer, intent(in) :: id
         real(dp), intent(in) :: y ! new value for average he4 mass fraction
         integer, intent(in) :: nzlo, nzhi ! change cells from nzlo to nzhi, inclusive.
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         call set_y(s, y, nzlo, nzhi, ierr)
      end subroutine set_y_in_section
      
      
      subroutine star_set_z(id, new_z, ierr)
         ! NOTE: this routine simply changes abundances; it doesn't reconverge the model.
         ! to make large changes in z, you'll need to spread it out over a number of steps
         ! in order to let the model adjust to the changes a small amount at a time.
         use adjust_xyz, only: set_z
         use star_def, only: star_info
         integer, intent(in) :: id
         real(dp), intent(in) :: new_z
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         call set_z(s, new_z, 1, s% nz, ierr)
      end subroutine star_set_z
      
      
      subroutine set_z_in_section(id, new_z, nzlo, nzhi, ierr)
         ! NOTE: this routine simply changes abundances; it doesn't reconverge the model.
         ! to make large changes in z, you'll need to spread it out over a number of steps
         ! in order to let the model adjust to the changes a small amount at a time.
         ! BTW: the set_z routine considers everything to be a "metal" except H1 and He4.
         use adjust_xyz, only: set_z
         use star_def, only: star_info
         integer, intent(in) :: id
         real(dp), intent(in) :: new_z
         integer, intent(in) :: nzlo, nzhi ! change cells from nzlo to nzhi, inclusive.
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         call set_z(s, new_z, nzlo, nzhi, ierr)
      end subroutine set_z_in_section
      
      
      subroutine star_replace_element(id, chem1, chem2, ierr)
         ! replaces chem1 by chem2.
         ! NOTE: this routine simply changes abundances; it doesn't reconverge the model.
         use star_def, only: star_info
         integer, intent(in) :: id
         integer, intent(in) :: chem1, chem2 ! values are chem_id's such as ihe4.  see chem_def.
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         call replace_element_in_section(id, chem1, chem2, 1, s% nz, ierr)
      end subroutine star_replace_element
      
      
      subroutine replace_element_in_section(id, chem1, chem2, nzlo, nzhi, ierr)
         ! replaces chem1 by chem2.
         ! NOTE: this routine simply changes abundances; it doesn't reconverge the model.
         use adjust_xyz, only: do_replace
         use star_def, only: star_info
         integer, intent(in) :: id
         integer, intent(in) :: chem1, chem2 ! values are chem_id's such as ihe4.  see chem_def.
         integer, intent(in) :: nzlo, nzhi ! change cells from nzlo to nzhi, inclusive.
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         call do_replace(s, chem1, chem2, nzlo, nzhi, ierr)
      end subroutine replace_element_in_section
      
      
      subroutine star_set_abundance(id, chem_id, new_frac, ierr)
         ! set mass fraction of species to new_frac uniformly in cells nzlo to nzhi
         ! 
         ! NOTE: this routine simply changes abundances; it doesn't reconverge the model.
         use star_def, only: star_info
         integer, intent(in) :: id
         integer, intent(in) :: chem_id ! a chem_id such as ihe4.  see chem_def.
         real(dp), intent(in) :: new_frac
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         call set_abundance_in_section(id, chem_id, new_frac, 1, s% nz, ierr)
      end subroutine star_set_abundance
      
      
      subroutine set_abundance_in_section(id, chem_id, new_frac, nzlo, nzhi, ierr)
         ! set mass fraction of species to new_frac uniformly in cells nzlo to nzhi
         ! NOTE: this routine simply changes abundances; it doesn't reconverge the model.
         use adjust_xyz, only: do_set_abundance
         use star_def, only: star_info
         integer, intent(in) :: id
         integer, intent(in) :: chem_id ! a chem_id such as ihe4.  see chem_def.
         real(dp), intent(in) :: new_frac
         integer, intent(in) :: nzlo, nzhi ! change cells from nzlo to nzhi, inclusive.
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         call do_set_abundance(s, chem_id, new_frac, nzlo, nzhi, ierr)
      end subroutine set_abundance_in_section
      
      
      subroutine uniform_mix_section(id, nzlo, nzhi, ierr)
         ! uniformly mix abundances in cells nzlo to nzhi
         ! NOTE: this routine simply changes abundances; it doesn't reconverge the model.
         use adjust_xyz, only: do_uniform_mix_section
         use star_def, only: star_info
         integer, intent(in) :: id
         integer, intent(in) :: nzlo, nzhi ! change cells from nzlo to nzhi, inclusive.
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         write(*,*) 'uniform_mix_section'
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         call do_uniform_mix_section(s, s% species, nzlo, nzhi, ierr)
      end subroutine uniform_mix_section
      
      
      subroutine uniform_mix_envelope_down_to_T(id, T, ierr)
         ! uniformly mix abundances in cells from surface down to given temperature
         ! NOTE: this routine simply changes abundances; it doesn't reconverge the model.
         use adjust_xyz, only: do_uniform_mix_envelope_down_to_T
         use star_def, only: star_info
         integer, intent(in) :: id
         real(dp), intent(in) :: T
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         write(*,*) 'uniform_mix_envelope_down_to_T'
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         call do_uniform_mix_envelope_down_to_T(s, T, ierr)
      end subroutine uniform_mix_envelope_down_to_T

 
      ! access to the value of the next timestep
      
      real(dp) function get_dt_next(id, ierr)
         use star_private_def
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         type (star_info), pointer :: s         
         call star_ptr(id, s, ierr)
         if (ierr /= 0) then
            get_dt_next = -1
            return
         end if
         get_dt_next = s% dt_next
      end function get_dt_next
      
      
      subroutine set_dt_next(dt, id, ierr)
         use star_private_def
         integer, intent(in) :: id
         real(dp), intent(in) :: dt
         integer, intent(out) :: ierr
         type (star_info), pointer :: s         
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         s% dt_next = dt
      end subroutine set_dt_next



      ! relaxation routines (for "pseudo-evolution" of the model)
      
      subroutine star_relax_mass(id, new_mass, lg_max_abs_mdot, ierr) ! also resets initial_mass
         ! acts like accretion or wind to change star mass
         use relax, only: do_relax_mass
         integer, intent(in) :: id
         real(dp), intent(in) :: new_mass ! in Msun units
         real(dp), intent(in) :: lg_max_abs_mdot ! in log10(Msun/year)
            ! e.g., -8.0 for mdot of -10^-8 Msun/year
         integer, intent(out) :: ierr
         call do_relax_mass(id, new_mass, lg_max_abs_mdot, ierr)      
      end subroutine star_relax_mass
      
      
      subroutine star_relax_mass_scale( &
            id, new_mass, dlgm_per_step, change_mass_years_for_dt, ierr) ! also resets initial_mass
         ! rescales star mass without changing composition as function of m/mstar
         use relax, only: do_relax_mass_scale
         integer, intent(in) :: id
         real(dp), intent(in) :: new_mass ! in Msun units
         real(dp), intent(in) :: dlgm_per_step, change_mass_years_for_dt
         integer, intent(out) :: ierr
         call do_relax_mass_scale(id, new_mass, dlgm_per_step, change_mass_years_for_dt, ierr)      
      end subroutine star_relax_mass_scale
      
      
      subroutine star_relax_core( &
            id, new_core_mass, dlg_core_mass_per_step, &
            relax_core_years_for_dt, core_avg_rho, core_avg_eps, ierr)
         use relax, only: do_relax_core
         integer, intent(in) :: id
         real(dp), intent(in) :: new_core_mass ! in Msun units
         real(dp), intent(in) :: dlg_core_mass_per_step, relax_core_years_for_dt
         real(dp), intent(in) :: core_avg_rho, core_avg_eps
            ! adjust R_center according to core_avg_rho (g cm^-3)
            ! adjust L_center according to core_avg_eps (erg g^-1 s^-1)
         integer, intent(out) :: ierr
         call do_relax_core( &
            id, new_core_mass, dlg_core_mass_per_step, &
            relax_core_years_for_dt, core_avg_rho, core_avg_eps, ierr)      
      end subroutine star_relax_core
      
      
      subroutine star_relax_M_center( &
            id, new_mass, dlgm_per_step, relax_M_center_dt, ierr)
         use relax, only: do_relax_M_center
         integer, intent(in) :: id
         real(dp), intent(in) :: new_mass ! in Msun units
         real(dp), intent(in) :: dlgm_per_step, relax_M_center_dt
         integer, intent(out) :: ierr
         call do_relax_M_center( &
            id, new_mass, dlgm_per_step, relax_M_center_dt, ierr)      
      end subroutine star_relax_M_center
      
      
      subroutine star_relax_R_center( &
            id, new_R_center, dlgR_per_step, relax_R_center_dt, ierr)
         use relax, only: do_relax_R_center
         integer, intent(in) :: id
         real(dp), intent(in) :: new_R_center ! in cm
         real(dp), intent(in) :: dlgR_per_step, relax_R_center_dt
         integer, intent(out) :: ierr
         call do_relax_R_center( &
            id, new_R_center, dlgR_per_step, relax_R_center_dt, ierr)      
      end subroutine star_relax_R_center
      
      
      subroutine star_relax_v_center( &
            id, new_v_center, dv_per_step, relax_v_center_dt, ierr)
         use relax, only: do_relax_v_center
         integer, intent(in) :: id
         real(dp), intent(in) :: new_v_center ! in cm/s
         real(dp), intent(in) :: dv_per_step, relax_v_center_dt
         integer, intent(out) :: ierr
         call do_relax_v_center( &
            id, new_v_center, dv_per_step, relax_v_center_dt, ierr)      
      end subroutine star_relax_v_center
      
      
      subroutine star_relax_L_center( &
            id, new_L_center, dlgL_per_step, relax_L_center_dt, ierr)
         use relax, only: do_relax_L_center
         integer, intent(in) :: id
         real(dp), intent(in) :: new_L_center ! in ergs/second
         real(dp), intent(in) :: dlgL_per_step, relax_L_center_dt
         integer, intent(out) :: ierr
         call do_relax_L_center( &
            id, new_L_center, dlgL_per_step, relax_L_center_dt, ierr)      
      end subroutine star_relax_L_center
      
      
      subroutine star_relax_dxdt_nuc_factor(id, new_value, per_step_multiplier, ierr)
         use relax, only: do_relax_dxdt_nuc_factor
         integer, intent(in) :: id
         real(dp), intent(in) :: new_value
         real(dp), intent(in) :: per_step_multiplier
         integer, intent(out) :: ierr
         call do_relax_dxdt_nuc_factor(id, new_value, per_step_multiplier, ierr)      
      end subroutine star_relax_dxdt_nuc_factor
      
      
      subroutine star_relax_eps_nuc_factor(id, new_value, per_step_multiplier, ierr)
         use relax, only: do_relax_eps_nuc_factor
         integer, intent(in) :: id
         real(dp), intent(in) :: new_value
         real(dp), intent(in) :: per_step_multiplier
         integer, intent(out) :: ierr
         call do_relax_eps_nuc_factor(id, new_value, per_step_multiplier, ierr)      
      end subroutine star_relax_eps_nuc_factor
      
      
      subroutine star_relax_opacity_max(id, new_value, per_step_multiplier, ierr)
         use relax, only: do_relax_opacity_max
         integer, intent(in) :: id
         real(dp), intent(in) :: new_value
         real(dp), intent(in) :: per_step_multiplier
         integer, intent(out) :: ierr
         call do_relax_opacity_max(id, new_value, per_step_multiplier, ierr)      
      end subroutine star_relax_opacity_max
      
      
      subroutine star_relax_max_surf_dq(id, new_value, per_step_multiplier, ierr)
         use relax, only: do_relax_max_surf_dq
         integer, intent(in) :: id
         real(dp), intent(in) :: new_value
         real(dp), intent(in) :: per_step_multiplier
         integer, intent(out) :: ierr
         call do_relax_max_surf_dq(id, new_value, per_step_multiplier, ierr)      
      end subroutine star_relax_max_surf_dq
      
      
      subroutine star_relax_composition( &
            id, num_steps_to_use, num_pts, species, xa, xq, ierr)
         ! with normal composition changes turned off,
         ! incrementally revise composition to get requested profile
         use relax, only: do_relax_composition
         integer, intent(in) :: id
         integer, intent(in) :: num_steps_to_use ! use this many steps to do conversion
         integer, intent(in) :: num_pts 
            ! length of composition vector; need not equal nz for current model (will interpolate)
         integer, intent(in) :: species 
            ! must = number of species for current model
         real(dp), intent(in) :: xa(:,:) ! (species, num_pts) ! target composition profile
         real(dp), intent(in) :: xq(:) ! (num_pts)
            ! xq(i) = fraction of xmstar exterior to the point i
            ! where xmstar = mstar - M_center
         integer, intent(out) :: ierr
         call do_relax_composition(id, num_steps_to_use, num_pts, species, xa, xq, ierr) 
      end subroutine star_relax_composition
      
      
      subroutine star_relax_to_xaccrete(id, num_steps_to_use, ierr)
         ! with normal composition changes turned off,
         ! incrementally revise composition to get uniform match to current accretion specs
         use relax, only: do_relax_to_xaccrete
         integer, intent(in) :: id
         integer, intent(in) :: num_steps_to_use ! use this many steps to do conversion
         integer, intent(out) :: ierr
         call do_relax_to_xaccrete(id, num_steps_to_use, ierr) 
      end subroutine star_relax_to_xaccrete

      
      subroutine star_relax_Y(id, new_Y, dY, ierr) ! also resets initial_y
         use relax, only: do_relax_Y
         integer, intent(in) :: id
         real(dp), intent(in) :: new_Y
         real(dp), intent(in) :: dY ! change Y by this amount per step
         integer, intent(out) :: ierr
         call do_relax_Y(id, new_Y, dY, ierr)      
      end subroutine star_relax_Y

      
      subroutine star_relax_Z(id, new_z, dlnz, minq, maxq, ierr) ! also resets initial_z
         use relax, only: do_relax_Z
         integer, intent(in) :: id
         real(dp), intent(in) :: new_z
         real(dp), intent(in) :: dlnz ! change lnz by this amount per step
         real(dp), intent(in) :: minq, maxq ! change in this q range
         integer, intent(out) :: ierr
         call do_relax_Z(id, new_z, dlnz, minq, maxq, ierr)      
      end subroutine star_relax_Z

      
      ! the optical depth of the outermost cell is tau_factor*tau_photosphere
      ! for normal hydrostatic stellar evolution, tau_factor = 1
      ! but in general, the limits are 0 < tau_factor <= 1,
      ! so by making tau_factor << 1, you can include the atmosphere in the model.
      subroutine star_relax_tau_factor(id, new_tau_factor, dlogtau_factor, ierr)
         use relax, only: do_relax_tau_factor
         integer, intent(in) :: id
         real(dp), intent(in) :: new_tau_factor
         real(dp), intent(in) :: dlogtau_factor
            ! change log10(tau_factor) by at most this amount per step
         integer, intent(out) :: ierr
         call do_relax_tau_factor(id, new_tau_factor, dlogtau_factor, ierr)      
      end subroutine star_relax_tau_factor      
      
      
      ! kind_of_relax = 0 => target = new_omega 
      ! kind_of_relax = 1 => target = new_omega_div_omega_crit 
      ! kind_of_relax = 2 => target = new_surface_rotation_v 
      subroutine star_relax_uniform_omega(id, &
            kind_of_relax, target_value, num_steps_to_relax_rotation, &
            relax_omega_max_yrs_dt, ierr)
         use relax, only: do_relax_uniform_omega
         integer, intent(in) :: id, kind_of_relax, num_steps_to_relax_rotation
         real(dp), intent(in) :: target_value,relax_omega_max_yrs_dt
         integer, intent(out) :: ierr
         call do_relax_uniform_omega(id, &
            kind_of_relax, target_value, num_steps_to_relax_rotation, &
            relax_omega_max_yrs_dt, ierr)      
      end subroutine star_relax_uniform_omega
      
      
      subroutine star_relax_irradiation(id, &
            min_steps, new_irrad_flux, new_irrad_col_depth, &
            relax_irradiation_max_yrs_dt, ierr)
         use relax, only: do_relax_irradiation
         integer, intent(in) :: id, min_steps
         real(dp), intent(in) :: &
            new_irrad_flux, new_irrad_col_depth, relax_irradiation_max_yrs_dt
         integer, intent(out) :: ierr
         call do_relax_irradiation(id, &
            min_steps, new_irrad_flux, new_irrad_col_depth, relax_irradiation_max_yrs_dt, ierr)      
      end subroutine star_relax_irradiation
      
      
      subroutine star_relax_mass_change( &
            id, min_steps, initial_mass_change, final_mass_change, relax_mass_change_max_yrs_dt, ierr)
         use relax, only: do_relax_mass_change
         integer, intent(in) :: id, min_steps
         real(dp), intent(in) :: initial_mass_change, final_mass_change, relax_mass_change_max_yrs_dt
         integer, intent(out) :: ierr
         call do_relax_mass_change( &
            id, min_steps, initial_mass_change, final_mass_change, relax_mass_change_max_yrs_dt, ierr)      
      end subroutine star_relax_mass_change
      
      
      subroutine star_relax_num_steps(id, num_steps, max_timestep, ierr)
         use relax, only: do_relax_num_steps
         integer, intent(in) :: id, num_steps
         real(dp), intent(in) :: max_timestep
         integer, intent(out) :: ierr
         call do_relax_num_steps(id, num_steps, max_timestep, ierr)      
      end subroutine star_relax_num_steps
      
            
      ! evolve until star_check_limits returns terminate.
      subroutine star_evolve_to_limit(id, restore_at_end, ierr)
         use relax, only: do_relax_to_limit
         integer, intent(in) :: id
         logical, intent(in) :: restore_at_end
         integer, intent(out) :: ierr
         call do_relax_to_limit(id, restore_at_end, ierr)      
      end subroutine star_evolve_to_limit
      
      
      ! evolve until check_model says to stop.
      ! this is intended for use in special "relax to" operations.
      ! for normal evolution, you will probably want to use the ./rn script.
      subroutine star_evolve_to_check_point( &
            id, before_evolve, check_model, finish_model, &
            restore_at_end, lipar, ipar, lrpar, rpar, ierr)
         use relax, only: do_internal_evolve
         integer, intent(in) :: id, lipar, lrpar
         logical, intent(in) :: restore_at_end
         integer, intent(inout), pointer :: ipar(:) ! (lipar)
         real(dp), intent(inout), pointer :: rpar(:) ! (lrpar)
         interface
            subroutine before_evolve(s, id, lipar, ipar, lrpar, rpar, ierr)
               use const_def, only: dp
               use star_def, only: star_info
               type (star_info), pointer :: s
               integer, intent(in) :: id, lipar, lrpar
               integer, intent(inout), pointer :: ipar(:) ! (lipar)
               real(dp), intent(inout), pointer :: rpar(:) ! (lrpar)
               integer, intent(out) :: ierr
            end subroutine before_evolve
            integer function check_model(s, id, lipar, ipar, lrpar, rpar)
               ! returns either keep_going, redo, retry, backup, or terminate.
               ! for okay termination, set s% termination_code = t_relax_finished_okay
               use const_def, only: dp
               use star_def, only: star_info
               type (star_info), pointer :: s
               integer, intent(in) :: id, lipar, lrpar
               integer, intent(inout), pointer :: ipar(:) ! (lipar)
               real(dp), intent(inout), pointer :: rpar(:) ! (lrpar)
            end function check_model
            integer function finish_model(s)
               use star_def, only:star_info
               type (star_info), pointer :: s
            end function finish_model
         end interface
         integer, intent(out) :: ierr
         call do_internal_evolve( &
            id, before_evolve, check_model, finish_model, &
            restore_at_end, lipar, ipar, lrpar, rpar, ierr)      
      end subroutine star_evolve_to_check_point
      
      
      ! I use this sometimes for debugging.
      subroutine star_special_test(id, ierr)
         use star_def, only: star_info
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
      end subroutine star_special_test
      
      
      
      ! rotation
      
      ! note: this applies to the current model only; 
      ! subsequenct models may evolve away from solid body rotation. 
      subroutine star_set_uniform_omega(id, omega, ierr)
         use hydro_rotation, only: set_uniform_omega
         integer, intent(in) :: id
         real(dp), intent(in) :: omega
         integer, intent(out) :: ierr
         call set_uniform_omega(id, omega, ierr)
      end subroutine star_set_uniform_omega

      
      ! write "internals" data file (for debugging)
      
      ! this routine outputs columns of data for the current model
      subroutine write_internals(id, filename, ierr)
         use report, only: write_internals_to_file
         integer, intent(in) :: id
         character (len=*), intent(in) :: filename
         integer, intent(out) :: ierr
         call write_internals_to_file(id, filename, ierr)
      end subroutine write_internals
      
      
      ! this creates a filename and then calls write_internals.
      ! e.g., if num = 0, the filename is 'plot_data/internals0.data'
      subroutine std_write_internals(id, num)
         use report, only: std_write_internals_to_file
         integer, intent(in) :: id
         integer, intent(in) :: num ! mod(num, 10) is used for the filename.
         call std_write_internals_to_file(id, num)
      end subroutine std_write_internals
      
      
      
      ! a few miscellaneous extra routines for special jobs

      
      ! get the hydrostatic equilibrium value for dlnP/dm at point k
      ! Ppoint is the pressure at point k
      ! multiply dlnP_dm_qhse by Ppoint to get dP_dm_qhse
      ! multiply dlnP_dm_qhse by s% gradT(k) to get dlnT_dm_qhse at k
      subroutine star_dlnPdm_qhse(id, k, dlnP_dm_qhse, Ppoint, ierr)
         use hydro_vars, only: eval_dlnPdm_qhse
         use star_def, only: star_info
         integer, intent(in) :: id, k
         real(dp), intent(out) :: dlnP_dm_qhse, Ppoint
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         real(dp) :: &
            m, d_dlnPdm_dlnR, d_dlnPdm_dL, &
            d_dlnPdm_dlnd00, d_dlnPdm_dlnT00, &
            d_dlnPdm_dlndm1, d_dlnPdm_dlnTm1, &
            d_dlnPdm_dlnPgas00_const_T, d_dlnPdm_dlnT00_const_Pgas, &
            d_dlnPdm_dlnPgasm1_const_T, d_dlnPdm_dlnTm1_const_Pgas, &
            dPpoint_dlnd00, dPpoint_dlndm1, dPpoint_dlnT00, dPpoint_dlnTm1, &
            dPpoint_dlnPgas00_const_T, dPpoint_dlnPgasm1_const_T, &
            dPpoint_dlnT00_const_Pgas, dPpoint_dlnTm1_const_Pgas
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         call eval_dlnPdm_qhse(s, k, &
            m, dlnP_dm_qhse, d_dlnPdm_dlnR, d_dlnPdm_dL, &
            d_dlnPdm_dlnd00, d_dlnPdm_dlnT00, &
            d_dlnPdm_dlndm1, d_dlnPdm_dlnTm1, &
            d_dlnPdm_dlnPgas00_const_T, d_dlnPdm_dlnT00_const_Pgas, &
            d_dlnPdm_dlnPgasm1_const_T, d_dlnPdm_dlnTm1_const_Pgas, &
            Ppoint, &
            dPpoint_dlnd00, dPpoint_dlndm1, dPpoint_dlnT00, dPpoint_dlnTm1, &
            dPpoint_dlnPgas00_const_T, dPpoint_dlnPgasm1_const_T, &
            dPpoint_dlnT00_const_Pgas, dPpoint_dlnTm1_const_Pgas, &
            ierr)
      end subroutine star_dlnPdm_qhse

      
      ! call this if you want a description of the terminal log output
      subroutine show_log_description(id, ierr)
         use do_one_utils, only: do_show_log_description
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         call do_show_log_description(id, ierr)
      end subroutine show_log_description
      
      
      ! write the terminal header lines
      subroutine show_terminal_header(id, ierr)
         use star_def, only: star_info
         use do_one_utils, only: do_show_terminal_header
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         call do_show_terminal_header(s)
      end subroutine show_terminal_header
      
      
      ! write the terminal summary lines
      subroutine write_terminal_summary(id, ierr)
         use star_def, only: star_info
         use do_one_utils, only: do_terminal_summary
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         call do_terminal_summary(s)
      end subroutine write_terminal_summary
      
      
      subroutine save_profile(id, id_extra, &
            how_many_extra_profile_columns, data_for_extra_profile_columns, &
            priority, ierr)
         use star_def, only: star_info
         use profile, only: do_save_profiles
         integer, intent(in) :: id, id_extra
         interface
            include 'extra_profile_cols.inc'
         end interface
         integer, intent(in) :: priority
            ! there is a limit to how many profiles are saved, 
            ! and lower priority models are discarded if necessary
            ! to make room for higher priority ones.
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         s% save_profiles_model_priority = priority
         call do_save_profiles(s, id_extra, &
            how_many_extra_profile_columns, data_for_extra_profile_columns, ierr)
      end subroutine save_profile
      
      
      subroutine star_write_profile_info(id, fname, id_extra, &
            how_many_extra_profile_columns, data_for_extra_profile_columns, ierr)
         use star_def, only: star_info
         use profile, only: write_profile_info
         integer, intent(in) :: id, id_extra
         character (len=*) :: fname
         interface
            include 'extra_profile_cols.inc'
         end interface
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         call write_profile_info(s, fname, id_extra, &
            how_many_extra_profile_columns, data_for_extra_profile_columns, ierr)
      end subroutine star_write_profile_info

      
      subroutine name_for_restart_file(val, photo_digits, num_string)
         integer, intent(in) :: val, photo_digits
         character (len=*), intent(out) :: num_string
         call string_for_model_number('x', val, photo_digits, num_string)
      end subroutine name_for_restart_file

  
      subroutine string_for_model_number(prefix, n, num_digits, num_string)
         use star_utils, only: get_string_for_model_number
         character (len=*), intent(in) :: prefix
         integer, intent(in) :: n, num_digits
         character (len=*), intent(out) :: num_string
         call get_string_for_model_number(prefix, n, num_digits, num_string)
      end subroutine string_for_model_number


      ! a lightweight replacement for star_check_model
      integer function bare_bones_check_model(id)
         use do_one_utils, only: do_bare_bones_check_model
         integer, intent(in) :: id
         bare_bones_check_model = do_bare_bones_check_model(id)
      end function bare_bones_check_model
      
      
      ! get a value using the profile column id to specify
      real(dp) function val_for_profile(s, c, k)
         use star_def, only: star_info
         use profile_getval, only: getval_for_profile
         type (star_info), pointer :: s
         integer, intent(in) :: c ! one of the values like p_logL defined in star_def
         integer, intent(in) :: k ! the zone number
         logical :: int_flag 
         integer :: int_val
         call getval_for_profile(s, c, k, val_for_profile, int_flag, int_val)
         if (int_flag) val_for_profile = dble(int_val)
      end function val_for_profile
      
      
      ! get number of zones in current model
      integer function star_zones(id, ierr)
         use star_def, only: star_info
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         call star_ptr(id, s, ierr)
         if (ierr /= 0) then
            star_zones = -1
            return
         end if
         star_zones = s% nz
      end function star_zones

      
      real(dp) function get_current_y(id, ierr)
         use star_def, only: star_info
         use star_utils, only: eval_current_y
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         call star_ptr(id, s, ierr)
         if (ierr /= 0) then
            get_current_y = -1
            return
         end if
         get_current_y = eval_current_y(s, 1, s% nz, ierr)
      end function get_current_y

      
      real(dp) function get_current_y_in_section(id, nzlo, nzhi, ierr)
         use star_def, only: star_info
         use star_utils, only: eval_current_y
         integer, intent(in) :: id
         integer, intent(in) :: nzlo, nzhi ! consider only zones nzlo to nzhi inclusive
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         call star_ptr(id, s, ierr)
         if (ierr /= 0) then
            get_current_y_in_section = -1
            return
         end if
         get_current_y_in_section = eval_current_y(s, nzlo, nzhi, ierr)
      end function get_current_y_in_section

      
      real(dp) function get_current_y_at_point(id, k, ierr)
         use star_def, only: star_info
         use star_utils, only: eval_current_y
         integer, intent(in) :: id
         integer, intent(in) :: k ! between 1 and nz
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         call star_ptr(id, s, ierr)
         if (ierr /= 0) then
            get_current_y_at_point = -1
            return
         end if
         get_current_y_at_point = eval_current_y(s, k, k, ierr)
      end function get_current_y_at_point

      
      real(dp) function get_current_z(id, ierr)
         use star_def, only: star_info
         use star_utils, only: eval_current_z
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         call star_ptr(id, s, ierr)
         if (ierr /= 0) then
            get_current_z = -1
            return
         end if
         get_current_z = eval_current_z(s, 1, s% nz, ierr)
      end function get_current_z

      
      real(dp) function get_current_z_in_section(id, nzlo, nzhi, ierr)
         use star_def, only: star_info
         use star_utils, only: eval_current_z
         integer, intent(in) :: id
         integer, intent(in) :: nzlo, nzhi ! consider only zones nzlo to nzhi inclusive
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         call star_ptr(id, s, ierr)
         if (ierr /= 0) then
            get_current_z_in_section = -1
            return
         end if
         get_current_z_in_section = eval_current_z(s, nzlo, nzhi, ierr)
      end function get_current_z_in_section

      
      real(dp) function get_current_z_at_point(id, k, ierr)
         use star_def, only: star_info
         use star_utils, only: eval_current_z
         integer, intent(in) :: id
         integer, intent(in) :: k ! between 1 and nz
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         call star_ptr(id, s, ierr)
         if (ierr /= 0) then
            get_current_z_at_point = -1
            return
         end if
         get_current_z_at_point = eval_current_z(s, k, k, ierr)
      end function get_current_z_at_point

      
      real(dp) function get_current_abundance(id, iso, ierr)
         ! returns mass fraction for iso
         use star_def, only: star_info
         use star_utils, only: eval_current_abundance
         integer, intent(in) :: id
         integer, intent(in) :: iso ! chem id from chem_def
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         call star_ptr(id, s, ierr)
         if (ierr /= 0) then
            get_current_abundance = -1
            return
         end if
         get_current_abundance = &
            eval_current_abundance(s, s% net_iso(iso), 1, s% nz, ierr)
      end function get_current_abundance

      
      real(dp) function current_abundance_in_section(id, iso, nzlo, nzhi, ierr)
         ! returns mass fraction for iso
         use star_def, only: star_info
         use star_utils, only: eval_current_abundance
         integer, intent(in) :: id
         integer, intent(in) :: iso ! chem id from chem_def
         integer, intent(in) :: nzlo, nzhi ! consider only zones nzlo to nzhi inclusive
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         call star_ptr(id, s, ierr)
         if (ierr /= 0) then
            current_abundance_in_section = -1
            return
         end if
         current_abundance_in_section = &
            eval_current_abundance(s, s% net_iso(iso), nzlo, nzhi, ierr)
      end function current_abundance_in_section

      
      real(dp) function current_abundance_at_point(id, iso, k, ierr)
         ! returns mass fraction for iso
         use star_def, only: star_info
         use star_utils, only: eval_current_abundance
         integer, intent(in) :: id
         integer, intent(in) :: iso ! chem id from chem_def
         integer, intent(in) :: k
         integer, intent(out) :: ierr
         current_abundance_at_point = current_abundance_in_section(id, iso, k, k, ierr)
      end function current_abundance_at_point
      
      
      subroutine star_get_XYZ(id, xa, X, Y, Z, ierr)
         use star_def, only: star_info
         use star_utils, only: get_XYZ
         integer, intent(in) :: id
         real(dp), intent(in) :: xa(:)
         real(dp), intent(out) :: X, Y, Z
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         call get_XYZ(s, xa, X, Y, Z)
      end subroutine star_get_XYZ


      subroutine star_xa_for_standard_metals( &
            s, species, chem_id, net_iso, &
            h1, h2, he3, he4, which_zfracs, &
            dump_missing_metals_into_heaviest, xa, ierr)
         use adjust_xyz, only: get_xa_for_standard_metals
         use star_def, only: star_info
         type (star_info), pointer :: s         
         integer, intent(in) :: species, chem_id(:), net_iso(:), which_zfracs
         real(dp), intent(in) :: h1, h2, he3, he4 ! mass fractions
         logical, intent(in) :: dump_missing_metals_into_heaviest
         real(dp), intent(out) :: xa(:) ! (species)
         integer, intent(out) :: ierr 
         call get_xa_for_standard_metals( &
            s, species, chem_id, net_iso, &
            h1, h2, he3, he4, which_zfracs, &
            dump_missing_metals_into_heaviest, xa, ierr)
      end subroutine star_xa_for_standard_metals
      
      
      subroutine star_info_at_q(s, q, &
            kbdy, m, r, lgT, lgRho, L, v, &
            lgP, g, X, Y, edv_H, edv_He, &
            scale_height, dlnX_dr, dlnY_dr, dlnRho_dr, &
            omega, omega_div_omega_crit)
         use star_def, only: star_info
         use report, only: get_info_at_q
         type (star_info), pointer :: s
         real(dp), intent(in) :: q ! relative mass coord
         integer, intent(out) :: kbdy
         real(dp), intent(out) :: &
            m, r, lgT, lgRho, L, v, &
            lgP, g, X, Y, edv_H, edv_He, &
            scale_height, dlnX_dr, dlnY_dr, dlnRho_dr, &
            omega, omega_div_omega_crit
         call get_info_at_q(s, q, &
            kbdy, m, r, lgT, lgRho, L, v, &
            lgP, g, X, Y, edv_H, edv_He, &
            scale_height, dlnX_dr, dlnY_dr, dlnRho_dr, &
            omega, omega_div_omega_crit)
      end subroutine star_info_at_q
      
      
      integer function get_model_number(id, ierr)
         use star_def, only: star_info
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         call star_ptr(id, s, ierr)
         if (ierr /= 0) then
            get_model_number = -1
            return
         end if
         get_model_number = s% model_number
      end function get_model_number
      
      
      logical function check_for_after_He_burn(s, he4_limit)
         use star_utils, only: after_He_burn
         use star_def, only: star_info
         type (star_info), pointer :: s   
         real(dp), intent(in) :: he4_limit      
         check_for_after_He_burn = after_He_burn(s, he4_limit)
      end function check_for_after_He_burn
      
      
      logical function check_for_after_C_burn(s, c12_limit)
         use star_utils, only: after_C_burn
         use star_def, only: star_info
         type (star_info), pointer :: s   
         real(dp), intent(in) :: c12_limit      
         check_for_after_C_burn = after_C_burn(s, c12_limit)
      end function check_for_after_C_burn
      
      
      ! intrinsic variables like T, Rho, kap, etc. are cell averages
      ! this routine returns an interpolated value at outer boundary of cell k
      real(dp) function star_interp_val_to_pt(v,k,sz,dq,debug_str)
         use star_utils, only: interp_val_to_pt
         integer, intent(in) :: k, sz
         real(dp), pointer :: v(:), dq(:) ! (sz)
         character (len=*), intent(in) :: debug_str
         star_interp_val_to_pt = interp_val_to_pt(v,k,sz,dq,debug_str)
      end function star_interp_val_to_pt


      ! this routine returns an interpolated value of xa(j,:) at outer boundary of cell k
      real(dp) function star_interp_xa_to_pt(xa,j,k,sz,dq,debug_str)
         use star_utils, only: interp_xa_to_pt
         real(dp), pointer :: xa(:,:), dq(:) ! (sz)
         integer, intent(in) :: j, k, sz
         character (len=*), intent(in) :: debug_str
         star_interp_xa_to_pt = interp_xa_to_pt(xa,j,k,sz,dq,debug_str)
      end function star_interp_xa_to_pt
      ! misc routines
      
      
      subroutine star_set_xqs(nz, xq, dq, ierr) ! set xq's using dq's
         use star_utils, only: set_xqs
         integer, intent(in) :: nz
         real(dp), intent(inout) :: dq(:) ! (nz)
         real(dp), intent(out) :: xq(:) ! (nz)
         integer, intent(out) :: ierr
         call set_xqs(nz, xq, dq, ierr)
      end subroutine star_set_xqs
      
      
      subroutine star_eos_get( &
            s, k, z, xh, abar, zbar, xa, &
            Rho, log10Rho, T, log10T, & 
            res, d_dlnd, d_dlnT, d_dabar, d_dzbar, ierr)
         use star_def, only: star_info
         use micro, only: eos_get
         use eos_def, only: num_eos_basic_results
         type (star_info), pointer :: s         
         integer, intent(in) :: k ! 0 means not being called for a particular cell
         real(dp), intent(in) :: &
            z, xh, abar, zbar, xa(:), Rho, log10Rho, T, log10T
         real(dp), dimension(num_eos_basic_results), intent(out) :: &
            res, d_dlnd, d_dlnT, d_dabar, d_dzbar
         integer, intent(out) :: ierr
         call eos_get( &
            s, k, z, xh, abar, zbar, xa, &
            Rho, log10Rho, T, log10T, & 
            res, d_dlnd, d_dlnT, d_dabar, d_dzbar, ierr)
      end subroutine star_eos_get
      
      
      real(dp) function star_get1_kap( &
            s, k, zbar, xa, q, Pgas_div_P, log10_rho, log10_T, &
            lnfree_e, d_lnfree_e_dlnRho, d_lnfree_e_dlnT, &
            frac_Type2, dlnkap_dlnd, dlnkap_dlnT, ierr)
         use opacities, only: get1_kap, fraction_of_op_mono
         use star_def, only: star_info
         type (star_info), pointer :: s
         integer, intent(in) :: k
         real(dp), intent(in) :: zbar, q, Pgas_div_P, log10_rho, log10_T, &
            lnfree_e, d_lnfree_e_dlnRho, d_lnfree_e_dlnT
         real(dp), intent(in), pointer :: xa(:)
         real(dp), intent(out) :: frac_Type2, dlnkap_dlnd, dlnkap_dlnT
         integer, intent(out) :: ierr
         star_get1_kap = get1_kap( &
            s, k, zbar, xa, q, Pgas_div_P, &
            log10_rho, log10_T, fraction_of_op_mono(s,k), &
            lnfree_e, d_lnfree_e_dlnRho, d_lnfree_e_dlnT, &
            frac_Type2, dlnkap_dlnd, dlnkap_dlnT, ierr)
      end function star_get1_kap
      
      
      integer function get_result_reason(id, ierr)
         use star_def, only: star_info
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         call star_ptr(id, s, ierr)
         if (ierr /= 0) then
            get_result_reason = -1
            return
         end if
         get_result_reason = s% result_reason
      end function get_result_reason
      
      
      real(dp) function eval_tau_at_r(id, r, ierr)
         ! optical depth tau at radius r (cm)
         ! r should be <= s% r(1) and >= s% Rcenter
         ! does linear interpolation wrt r within cell
         use star_def, only: star_info
         use star_utils, only: get_tau_at_r
         integer, intent(in) :: id
         real(dp), intent(in) :: r
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         call star_ptr(id, s, ierr)
         if (ierr /= 0) then
            eval_tau_at_r = -1
            return
         end if
         eval_tau_at_r = get_tau_at_r(s, r)
      end function eval_tau_at_r
      
      
      real(dp) function eval_total_times(id, ierr)
         use star_def, only: star_info
         use star_utils, only: total_times
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         call star_ptr(id, s, ierr)
         if (ierr /= 0) then
            eval_total_times = -1
            return
         end if
         eval_total_times = total_times(s)
      end function eval_total_times
      
      
      real(dp) function star_surface_omega_crit(id, ierr)
         use star_def, only: star_info
         use star_utils, only: set_surf_avg_rotation_info
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         call star_ptr(id, s, ierr)
         if (ierr /= 0) then
            star_surface_omega_crit = -1
            return
         end if
         call set_surf_avg_rotation_info(s)
         star_surface_omega_crit = s% omega_crit_avg_surf
      end function star_surface_omega_crit
      
      
      ! some routines for "stellar engineering"

      
      subroutine star_adjust_gradT_fraction(id, k, f)
         ! replace gradT by combo of grada_at_face and gradr
         ! s% gradT(k) = f*s% grada_at_face(k) + (1-f)*s% gradr(k)
         ! and fixup other info that depends on this (e.g. partials)
         use star_def, only: star_info
         use mlt_info, only: adjust_gradT_fraction
         integer, intent(in) :: id, k        
         real(dp), intent(in) :: f
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) then
            return
         end if
         call adjust_gradT_fraction(s, k, f)
      end subroutine star_adjust_gradT_fraction

      
      ! call this if you change gradT
      subroutine star_adjust_gradT_excess(id, k)
         ! adjust efficiency of energy transport
         use star_def, only: star_info
         use mlt_info, only: adjust_gradT_excess
         integer, intent(in) :: id, k        
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) then
            return
         end if
         call adjust_gradT_excess(s, k)
      end subroutine star_adjust_gradT_excess
      
      
      subroutine star_remove_center_at_cell_k(id, k, ierr)
         use init, only: do_remove_center_at_cell_k
         integer, intent(in) :: id, k
         integer, intent(out) :: ierr
         call do_remove_center_at_cell_k(id, k, ierr)      
      end subroutine star_remove_center_at_cell_k
      
      
      subroutine star_remove_center_by_temperature(id, temperature, ierr)
         use init, only: do_remove_center_by_temperature
         integer, intent(in) :: id
         real(dp), intent(in) :: temperature
         integer, intent(out) :: ierr
         call do_remove_center_by_temperature(id, temperature, ierr)      
      end subroutine star_remove_center_by_temperature
      
      
      subroutine star_remove_center_by_radius_cm(id, r_cm, ierr)
         use init, only: do_remove_center_by_radius_cm
         integer, intent(in) :: id
         real(dp), intent(in) :: r_cm
         integer, intent(out) :: ierr
         call do_remove_center_by_radius_cm(id, r_cm, ierr)      
      end subroutine star_remove_center_by_radius_cm
      
      
      subroutine star_remove_center_by_mass_fraction_q(id, q, ierr)
         use init, only: do_remove_inner_fraction_q
         integer, intent(in) :: id
         real(dp), intent(in) :: q
         integer, intent(out) :: ierr
         call do_remove_inner_fraction_q(id, q, ierr)      
      end subroutine star_remove_center_by_mass_fraction_q
      
      
      subroutine star_remove_center_by_mass_gm(id, m, ierr)
         use init, only: do_remove_center_by_mass_gm
         integer, intent(in) :: id
         real(dp), intent(in) :: m
         integer, intent(out) :: ierr
         call do_remove_center_by_mass_gm(id, m, ierr)      
      end subroutine star_remove_center_by_mass_gm

      
      subroutine smooth_abundances_in_section(id, cnt, nzlo, nzhi, ierr)
         ! purely for cosmetic purposes.  doesn't even try to conserve abundances.
         use star_def, only: star_info
         use star_utils, only: smooth_abundances
         integer, intent(in) :: id
         integer, intent(in) :: cnt ! make this many passes
         integer, intent(in) :: nzlo, nzhi ! only smooth zones nzlo to nzhi inclusive
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         call smooth_abundances(s, cnt, nzlo, nzhi, ierr)
      end subroutine smooth_abundances_in_section
      
      
      ! PGSTAR interface
      subroutine start_new_run_for_pgstar(s, ierr) ! reset logs
         use star_def, only: star_info
         use pgstar
         type (star_info), pointer :: s
         integer, intent(out) :: ierr
         call do_start_new_run_for_pgstar(s, ierr)
      end subroutine start_new_run_for_pgstar
      
      
      subroutine restart_run_for_pgstar(s, ierr)
         use star_def, only: star_info
         use pgstar
         type (star_info), pointer :: s
         integer, intent(out) :: ierr
         call do_restart_run_for_pgstar(s, ierr)
      end subroutine restart_run_for_pgstar


      subroutine read_pgstar_controls(s, ierr)
         use star_def, only: star_info
         use pgstar, only: do_read_pgstar_controls
         type (star_info), pointer :: s
         integer, intent(out) :: ierr
         call do_read_pgstar_controls(s, 'inlist', ierr)
      end subroutine read_pgstar_controls


      subroutine read_pgstar_inlist(s, inlist_fname, ierr)
         use star_def, only: star_info
         use pgstar, only: do_read_pgstar_controls
         type (star_info), pointer :: s
         character(*), intent(in) :: inlist_fname
         integer, intent(out) :: ierr
         call do_read_pgstar_controls(s, inlist_fname, ierr)
      end subroutine read_pgstar_inlist


      subroutine update_pgstar_plots( &
            s, must_write_files, id_extra, &
            how_many_extra_history_columns, &
            data_for_extra_history_columns, &
            how_many_extra_profile_columns, &
            data_for_extra_profile_columns, &
            ierr)
         use star_def, only: star_info
         use pgstar
         type (star_info), pointer :: s
         logical, intent(in) :: must_write_files
         integer, intent(in) :: id_extra
         interface
            include 'extra_profile_cols.inc'
            include 'extra_history_cols.inc'
         end interface
         integer, intent(out) :: ierr
         call do_pgstar_plots( &
            s, must_write_files, id_extra, &
            how_many_extra_history_columns, &
            data_for_extra_history_columns, &
            how_many_extra_profile_columns, &
            data_for_extra_profile_columns, &
            ierr)
      end subroutine update_pgstar_plots
      
      
      subroutine create_pgstar_file_name(s, dir, prefix, name)
         use star_def, only: star_info
         use pgstar, only: do_create_file_name
         type (star_info), pointer :: s
         character (len=*), intent(in) :: dir, prefix
         character (len=*), intent(out) :: name
         call do_create_file_name(s, dir, prefix, name)
      end subroutine create_pgstar_file_name
      

      subroutine pgstar_write_plot_to_file(s, p, filename, ierr)
         use star_def, only: star_info, pgstar_win_file_data
         use pgstar, only: do_write_plot_to_file
         type (star_info), pointer :: s
         type (pgstar_win_file_data), pointer :: p
         character (len=*), intent(in) :: filename
         integer, intent(out) :: ierr
         call do_write_plot_to_file(s, p, filename, ierr)
      end subroutine pgstar_write_plot_to_file
      
      
      subroutine set_pgstar_xaxis_bounds( &
            s, xaxis_by, win_xmin_in, win_xmax_in, xmargin, &
            xvec, xmin, xmax, xleft, xright, dx, &
            grid_min, grid_max, npts, ierr)
         use star_def, only: star_info
         use pgstar, only: do_set_xaxis_bounds
         type (star_info), pointer :: s
         character (len=*), intent(in) :: xaxis_by
         real, intent(in) :: win_xmin_in, win_xmax_in, xmargin
         real, pointer, dimension(:) :: xvec
         real, intent(out) :: xmin, xmax, xleft, xright, dx
         integer, intent(out) :: grid_min, grid_max, npts
         integer, intent(out) :: ierr
         call do_set_xaxis_bounds( &
            s, xaxis_by, win_xmin_in, win_xmax_in, xmargin, &
            xvec, xmin, xmax, xleft, xright, dx, &
            grid_min, grid_max, npts, ierr)
      end subroutine set_pgstar_xaxis_bounds
      
      
      subroutine show_pgstar_xaxis_by(s,by,ierr)
         use star_def, only: star_info
         use pgstar, only: do_show_xaxis_by
         type (star_info), pointer :: s
         character (len=*), intent(in) :: by
         integer, intent(out) :: ierr
         call do_show_xaxis_by(s,by,ierr)
      end subroutine show_pgstar_xaxis_by
      
      
      subroutine show_pgstar_annotations( &
            s, show_annotation1, show_annotation2, show_annotation3)
         use star_def, only: star_info
         use pgstar, only: do_show_pgstar_annotations
         type (star_info), pointer :: s
         logical, intent(in) :: &
            show_annotation1, show_annotation2, show_annotation3
         call do_show_pgstar_annotations( &
            s, show_annotation1, show_annotation2, show_annotation3)
      end subroutine show_pgstar_annotations      
      
      
      subroutine pgstar_show_box(s, str1, str2)
         use star_def, only: star_info
         use pgstar, only: show_box_pgstar
         type (star_info), pointer :: s
         character (len=*), intent(in) :: str1, str2
         call show_box_pgstar(s, str1, str2)
      end subroutine pgstar_show_box
      
      
      subroutine pgstar_show_title(s, title, pad)
         use star_def, only: star_info
         use pgstar, only: show_title_pgstar
         type (star_info), pointer :: s
         character (len=*), intent(in) :: title
         real, intent(in) :: pad
         optional pad
         real :: pad_arg
         pad_arg = 0
         if (present(pad)) pad_arg = pad
         call show_title_pgstar(s, title, pad_arg)
      end subroutine pgstar_show_title
      
      
      subroutine pgstar_show_xaxis_label(s, label, pad)
         use star_def, only: star_info
         use pgstar, only: show_xaxis_label_pgstar
         type (star_info), pointer :: s
         character (len=*), intent(in) :: label
         real, intent(in) :: pad
         optional pad
         real :: pad_arg
         pad_arg = 0
         if (present(pad)) pad_arg = pad
         call show_xaxis_label_pgstar(s, label, pad_arg)
      end subroutine pgstar_show_xaxis_label
      
      
      subroutine pgstar_show_left_yaxis_label(s, label, pad)
         use star_def, only: star_info
         use pgstar, only: show_left_yaxis_label_pgstar
         type (star_info), pointer :: s
         character (len=*), intent(in) :: label
         real, intent(in) :: pad
         optional pad
         real :: pad_arg
         pad_arg = 0
         if (present(pad)) pad_arg = pad
         call show_left_yaxis_label_pgstar(s, label, pad_arg)
      end subroutine pgstar_show_left_yaxis_label
      
      
      subroutine pgstar_show_right_yaxis_label(s, label, pad)
         use star_def, only: star_info
         use pgstar, only: show_right_yaxis_label_pgstar
         type (star_info), pointer :: s
         character (len=*), intent(in) :: label
         real, intent(in) :: pad
         optional pad
         real :: pad_arg
         pad_arg = 0
         if (present(pad)) pad_arg = pad
         call show_right_yaxis_label_pgstar(s, label, pad_arg)
      end subroutine pgstar_show_right_yaxis_label
      
      
      subroutine pgstar_show_left_axis_label_pgmtxt( &
            s, coord, fjust, label, pad)
         use star_def, only: star_info
         use pgstar, only: show_left_yaxis_label_pgmtxt_pgstar
         type (star_info), pointer :: s
         character (len=*), intent(in) :: label
         real, intent(in) :: pad, coord, fjust
         optional pad
         real :: pad_arg
         pad_arg = 0
         if (present(pad)) pad_arg = pad
         call show_left_yaxis_label_pgmtxt_pgstar( &
            s, coord, fjust, label, pad)
      end subroutine pgstar_show_left_axis_label_pgmtxt
      
      
      subroutine pgstar_show_right_axis_label_pgmtxt( &
            s, coord, fjust, label, pad)
         use star_def, only: star_info
         use pgstar, only: show_right_yaxis_label_pgmtxt_pgstar
         type (star_info), pointer :: s
         character (len=*), intent(in) :: label
         real, intent(in) :: pad, coord, fjust
         optional pad
         real :: pad_arg
         pad_arg = 0
         if (present(pad)) pad_arg = pad
         call show_right_yaxis_label_pgmtxt_pgstar( &
            s, coord, fjust, label, pad)
      end subroutine pgstar_show_right_axis_label_pgmtxt
      
      
      subroutine pgstar_show_model_number(s)
         use star_def, only: star_info
         use pgstar, only: show_model_number_pgstar
         type (star_info), pointer :: s
         call show_model_number_pgstar(s)
      end subroutine pgstar_show_model_number
      
      
      subroutine pgstar_show_age(s)
         use star_def, only: star_info
         use pgstar, only: show_age_pgstar
         type (star_info), pointer :: s
         call show_age_pgstar(s)
      end subroutine pgstar_show_age
      

      subroutine star_history_specs(s, num, names, specs, report)
         use star_def, only: star_info
         use history, only: get_history_specs
         type (star_info), pointer :: s
         integer, intent(in) :: num
         character (len=*), intent(in) :: names(:)
         integer, intent(out) :: specs(:)
         logical, intent(in) :: report
	      call get_history_specs(s, num, names, specs, report)
	   end subroutine star_history_specs
	   
	   
      subroutine star_history_values(s, num, specs, &
            is_int_value, int_values, values, failed_to_find_value)
         use star_def, only: star_info
         use history, only: get_history_values
         type (star_info), pointer :: s
         integer, intent(in) :: num
         integer, intent(in) :: specs(:)
         logical, intent(out) :: is_int_value(:)
         integer, intent(out) :: int_values(:)
         real(dp), intent(out) :: values(:)
         logical, intent(out) :: failed_to_find_value(:)
	      call get_history_values(s, num, specs, &
            is_int_value, int_values, values, failed_to_find_value)
      end subroutine star_history_values


      integer function star_get_profile_id(s,name)
         use star_def, only: star_info
         use profile, only : get_profile_id
         type (star_info), pointer :: s
         character(len=*),intent(in) :: name
         star_get_profile_id = get_profile_id(s,name)
         if (star_get_profile_id < 0) THEN
            write(*,*) "FATAL ERROR Bad value for profile name ",trim(name)
            stop 'star_get_profile_id'
         end if
      end function star_get_profile_id


      real(dp) function star_get_profile_val(s,id,k)
         use star_def, only: star_info
         use profile, only : get_profile_val
         type (star_info), pointer :: s
         integer,intent(in) :: id,k
         star_get_profile_val = get_profile_val(s,id,k)
      end function star_get_profile_val


      real(dp) function star_get_profile_output(s,name,k)
         use star_def, only: star_info
         use profile, only : get_profile_val
         type (star_info), pointer :: s
         character(len=*),intent(in) :: name
         integer,intent(in) :: k
         star_get_profile_output = get_profile_val(s,star_get_profile_id(s,name),k)
      end function star_get_profile_output


      real(dp) function star_get_history_output(s,name)
         use star_def, only: star_info
         use history, only: get_history_specs, get_history_values, get1_hist_value
         type (star_info), pointer :: s	
         character(len=*),intent(in) :: name
         integer,parameter :: num_rows=1
         real(dp) :: values(num_rows)
         integer :: int_values(num_rows), specs(num_rows)
         logical :: is_int_value(num_rows)
         logical :: failed_to_find_value(num_rows)
         call get_history_specs(s, num_rows, (/name/), specs, .false.)
         call get_history_values( &
            s, num_rows, specs, &
            is_int_value, int_values, values, failed_to_find_value)
         if (failed_to_find_value(num_rows)) then
            if (.not. get1_hist_value(s, name, values(num_rows))) then
               write(*,*) "FATAL ERROR Bad value for history name ",trim(name)
               stop 'star_get_history_output'
            end if
         end if
         if (is_int_value(1)) then 
            star_get_history_output=dble(int_values(num_rows))
         else
            star_get_history_output=values(num_rows)
         end if
      end function star_get_history_output
      
      subroutine set_rate_factors_from_categories(id, ierr)
         use net_def, only: Net_General_Info, get_net_ptr
			use star_def, only: star_info,get_star_ptr
			integer, intent(in) :: id
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
			integer :: j,ind
         type (Net_General_Info), pointer :: g
         ierr = 0
         call get_star_ptr(id, s, ierr)
         if (ierr /= 0) return
      
			ierr = 0
         call get_net_ptr(s%net_handle, g, ierr)

			do j=1,s%num_reactions
				ind=g% reaction_kind(j)
				s% rate_factors(j)=s% category_factors(ind)
			end do

      end subroutine set_rate_factors_from_categories

      end module star_lib
