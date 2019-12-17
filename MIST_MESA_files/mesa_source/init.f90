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

      module init

      use star_private_def
      use const_def

      implicit none
      
      integer, parameter :: do_create_pre_ms_model = 0
      integer, parameter :: do_load_zams_model = 1
      integer, parameter :: do_load_saved_model = 2
      integer, parameter :: do_create_initial_model = 3
      
      logical :: have_done_starlib_init = .false.
      

      contains
      
      
      subroutine set_kap_and_eos_handles(id, ierr)
         use kap_lib, only: alloc_kap_handle
         use eos_lib, only: alloc_eos_handle
         integer, intent(in) :: id
         integer, intent(out) :: ierr ! 0 means AOK.
         type (star_info), pointer :: s         
         call get_star_ptr(id, s, ierr)
         if (ierr /= 0) then
            write(*,*) 'set_kap_and_eos_handles failed in get_star_ptr'
            return
         end if
         if (s% eos_handle == 0) then
            s% eos_handle = alloc_eos_handle(ierr)
            if (ierr /= 0) then
               write(*,*) 'set_kap_and_eos_handles failed in alloc_eos_handle'
               return
            end if
         end if
         if (s% kap_handle == 0) then
            s% kap_handle = alloc_kap_handle(ierr)
            if (ierr /= 0) then
               write(*,*) 'set_kap_and_eos_handles failed in alloc_kap_handle'
               return
            end if
         end if
      end subroutine set_kap_and_eos_handles
      
      
      subroutine do_star_init( &
            my_mesa_dir, chem_isotopes_filename, &
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
         use paquette_coeffs, only: initialise_collision_integrals
         use hydro_rotation, only: init_rotation
         use alloc, only: init_alloc
         character (len=*), intent(in) :: &
            my_mesa_dir, chem_isotopes_filename, net_reaction_filename, &
            jina_reaclib_filename, rate_tables_dir, &
            kappa_file_prefix, kappa_CO_prefix, kappa_lowT_prefix, &
            eosDT_Z1_suffix, eosPT_Z1_suffix, &
            eos_file_prefix, rates_cache_suffix, &
            ionization_file_prefix, ionization_Z1_suffix, &
            eosDT_cache_dir, eosPT_cache_dir, eosDE_cache_dir, &
            ionization_cache_dir, kap_cache_dir, rates_cache_dir
         real(dp), intent(in) :: kappa_blend_logT_upper_bdy, kappa_blend_logT_lower_bdy, &
            kappa_type2_logT_lower_bdy
         integer, intent(out) :: ierr
         ! ierr will be 0 for a normal return. 
         ! ierr nonzero means something went wrong.
         integer :: iam, nprocs, nprow, npcol
         include 'formats'
         ierr = 0
         if (have_done_starlib_init) return
         call initialise_collision_integrals
         call init_alloc
         if (ierr /= 0) then
            write(*,*) 'superlu_dist_before returned ierr', ierr
            return
         end if

         call stardata_init( &
            my_mesa_dir, chem_isotopes_filename, &
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
      	if (ierr /= 0) then
            write(*,*) 'failed in stardata_init'
            return
      	end if
      	
         call init_rotation(ierr)
      	if (ierr /= 0) then
            write(*,*) 'failed in init_rotation'
            return
      	end if
      	
      	have_done_starlib_init = .true.
      	
      end subroutine do_star_init


      subroutine do_starlib_shutdown
         !use mtx_lib, only: superlu_dist_quit_work, superlu_dist_after
         !use micro, only: shutdown_microphys
         !integer :: ierr
         !call superlu_dist_quit_work(ierr)  
         !call superlu_dist_after(ierr)
         !call shutdown_microphys ! skip this for now
      end subroutine do_starlib_shutdown
            
      
      integer function alloc_star_data(ierr)
         use kap_lib
         use eos_lib
         use rates_def, only: rates_reaction_id_max, rates_NACRE_if_available
         use chem_def, only: num_categories
         use net, only: default_set_which_rates, default_set_rate_factors
         
         
         integer, intent(out) :: ierr
         
         type (star_info), pointer :: s
         integer, parameter :: init_alloc_nvar = 20
         character (len=32) :: extra_name
         integer :: i
                  
         ierr = 0
         
         alloc_star_data = alloc_star(ierr)
         if (ierr /= 0) then
            write(*,*) 'alloc_star_data failed in alloc_star'
            return
         end if
         
         call get_star_ptr(alloc_star_data, s, ierr)
         if (ierr /= 0) then
            write(*,*) 'alloc_star_data failed in get_star_ptr'
            return
         end if
         
         nullify(s% dq)
         nullify(s% dq_old)
         nullify(s% dq_older)

         nullify(s% lnT)
         nullify(s% lnT_old)
         nullify(s% lnT_older)

         nullify(s% conv_vel)
         nullify(s% conv_vel_old)
         nullify(s% conv_vel_older)

         nullify(s% nu_ST)
         nullify(s% nu_ST_old)
         nullify(s% nu_ST_older)

         nullify(s% D_DSI)
         nullify(s% D_DSI_old)
         nullify(s% D_DSI_older)

         nullify(s% D_SH)
         nullify(s% D_SH_old)
         nullify(s% D_SH_older)

         nullify(s% D_SSI)
         nullify(s% D_SSI_old)
         nullify(s% D_SSI_older)

         nullify(s% D_ES)
         nullify(s% D_ES_old)
         nullify(s% D_ES_older)

         nullify(s% D_GSF)
         nullify(s% D_GSF_old)
         nullify(s% D_GSF_older)

         nullify(s% D_mix_non_rotation)
         nullify(s% D_mix_old)
         nullify(s% D_mix_older)

         nullify(s% q)
         nullify(s% q_old)
         nullify(s% q_older)

         nullify(s% xa)
         nullify(s% xa_old)
         nullify(s% xa_older)

         nullify(s% xh)
         nullify(s% xh_old)
         nullify(s% xh_older)

         nullify(s% xa0_1, s% xa1_1, s% xa2_1, s% xa3_1, &
            s% xa4_1, s% xa5_1, s% xa6_1, s% xa_compare_1)
         nullify(s% xh0_1, s% xh1_1, s% xh2_1, s% xh3_1, &
            s% xh4_1, s% xh5_1, s% xh6_1, s% xh_compare_1)

         nullify(s% xa0, s% xa1, s% xa2, s% xa3, &
            s% xa4, s% xa5, s% xa6, s% xa_compare)
         nullify(s% xh0, s% xh1, s% xh2, s% xh3, &
            s% xh4, s% xh5, s% xh6, s% xh_compare)
         
         nullify( &
            s% op_mono_umesh1, s% op_mono_ff1, &
            s% op_mono_rs1, s% op_mono_s1)

         nullify(s% atm_structure)
         s% atm_structure_num_pts = 0
         
         nullify(s% chem_id)
         nullify(s% xa_removed)

         nullify(s% which_rates)
         s% set_which_rates => default_set_which_rates
         
         nullify(s% rate_factors)
         s% set_rate_factors => default_set_rate_factors
         
         allocate(s% nameofvar(init_alloc_nvar),stat=ierr)
         if (ierr /= 0) return
         
         allocate(s% nameofequ(init_alloc_nvar),stat=ierr)
         if (ierr /= 0) return
         
         allocate(s% ode_var(init_alloc_nvar),stat=ierr)
         if (ierr /= 0) return

         allocate(s% category_factors(num_categories),stat=ierr)
         if (ierr /= 0) return
         s% category_factors(:)=1.0
         
         do i = 1, max_num_profile_extras
            if (i < 10) then
               write(extra_name,'(a,i1)') 'profile_extra_', i
            else if (i < 100) then
               write(extra_name,'(a,i2)') 'profile_extra_', i
            else
               write(extra_name,'(a,i3)') 'profile_extra_', i
            end if
            s% profile_extra_name(i) = trim(extra_name)
         end do

         nullify(s% history_values)
         nullify(s% history_value_is_integer)
         nullify(s% history_names)
         nullify(s% history_names_dict)
 
         nullify(s% prev_mesh_xm)
         nullify(s% prev_mesh_lnS)
         nullify(s% prev_lnS)
         nullify(s% prev_lnS_const_q)
         nullify(s% prev_mesh_mu)
         nullify(s% prev_mu)
         
         nullify(s% other_star_info)
         
         nullify(s% bcyclic_sprs_storage)
         nullify(s% bcyclic_sprs_shared_ia)
         nullify(s% bcyclic_sprs_shared_ja)

         nullify(s% bcyclic_odd_storage)
         nullify(s% bcyclic_odd_storage_qp)

         nullify(s% ipar_decsol)
         nullify(s% hydro_iwork)
         nullify(s% rpar_decsol)
         nullify(s% hydro_work)
         
         s% using_ode_form = .false.
         nullify(s% ode1)
         nullify(s% ode)
         
         s% net_name = ''
         s% species = 0
         s% num_reactions = 0
         nullify(s% AF1)
         s% ebdf_stage = -1
         
         s% M_center = 0
         s% R_center = 0
         s% v_center = 0
         s% L_center = 0
         
         nullify(s% profile_column_spec)
         nullify(s% history_column_spec)
         
         s% num_conv_boundaries = 0
         nullify(s% conv_bdy_q)
         nullify(s% conv_bdy_loc)
         nullify(s% top_conv_bdy)
         
         s% num_mix_boundaries = 0
         nullify(s% mix_bdy_q)
         nullify(s% mix_bdy_loc)
         nullify(s% top_mix_bdy)
         
         nullify(s% burn_h_conv_region)
         nullify(s% burn_he_conv_region)
         nullify(s% burn_z_conv_region)
         
         s% have_burner_storage = .false.
         s% burner_storage_sz_per_thread = 0
         s% burner_num_threads = 0
         nullify(s% burner_storage)
                  
         s% doing_timing = .false.
         s% time_evolve_step = 0
         s% time_remesh = 0
         s% time_adjust_mass = 0
         s% time_element_diffusion = 0
         s% time_semi_explicit_hydro = 0
         s% time_struct_burn_mix = 0
         s% time_newton_matrix = 0
         s% time_solve_mix = 0
         s% time_solve_burn = 0
         s% time_solve_omega_mix = 0
         s% time_eos = 0
         s% time_neu_kap = 0
         s% time_nonburn_net = 0
         
         s% time_total = 0
         
         s% model_profile_filename = ''
         s% most_recent_profile_filename = ''
         
         s% model_controls_filename = ''
         s% most_recent_controls_filename = ''
         
         s% most_recent_photo_name = ''
         
         s% doing_flash_wind = .false.
         s% doing_rlo_wind = .false.
         s% doing_nova_wind = .false.

         s% phase_of_evolution = phase_starting
         s% recent_log_header = -1000
         s% post_he_age = -1d0
         s% profile_age = -1d0
         s% prev_cntr_rho = 1d99
         s% helium_ignition = .false.
         s% carbon_ignition = .false.
         
         s% tau_base = 2d0/3d0
         s% tau_factor = 1

         s% hydro_matrix_type = -1
         
         s% TP_state = 0
         s% have_done_TP = .false.

         s% using_revised_net_name = .false.
         s% revised_net_name = ''
         s% revised_net_name_old = ''
         s% revised_net_name_older = ''

         s% using_revised_max_yr_dt = .false.
         s% revised_max_yr_dt = 0
         s% revised_max_yr_dt_old = 0
         s% revised_max_yr_dt_older = 0

         s% astero_using_revised_max_yr_dt = .false.
         s% astero_revised_max_yr_dt = 0
         s% astero_revised_max_yr_dt_old = 0
         s% astero_revised_max_yr_dt_older = 0

         s% startup_increment_ebdf_order = .false.
         s% truncation_ratio = 0

         s% ebdf_order = 0
         s% ebdf_order_old = 0
         s% ebdf_order_older = 0

         s% ebdf_hold = 0
         s% ebdf_hold_old = 0
         s% ebdf_hold_older = 0

         s% cumulative_acoustic_L = 0
         s% cumulative_acoustic_L_old = 0
         s% cumulative_acoustic_L_older = 0

         s% cumulative_acoustic_L_center = 0
         s% cumulative_acoustic_L_center_old = 0
         s% cumulative_acoustic_L_center_older = 0

         s% cumulative_visc_heat_added = 0
         s% cumulative_visc_heat_added_old = 0
         s% cumulative_visc_heat_added_older = 0

         s% cumulative_eps_grav = 0
         s% cumulative_eps_grav_old = 0
         s% cumulative_eps_grav_older = 0

         s% cumulative_energy_error = 0
         s% cumulative_energy_error_old = 0
         s% cumulative_energy_error_older = 0

         s% cumulative_L_surf = 0
         s% cumulative_L_surf_old = 0
         s% cumulative_L_surf_older = 0

         s% cumulative_L_center = 0
         s% cumulative_L_center_old = 0
         s% cumulative_L_center_older = 0

         s% cumulative_extra_heating = 0
         s% cumulative_extra_heating_old = 0
         s% cumulative_extra_heating_older = 0

         s% cumulative_irradiation_heating = 0
         s% cumulative_irradiation_heating_old = 0
         s% cumulative_irradiation_heating_older = 0

         s% cumulative_nuclear_heating = 0
         s% cumulative_nuclear_heating_old = 0
         s% cumulative_nuclear_heating_older = 0

         s% cumulative_non_nuc_neu_cooling = 0
         s% cumulative_non_nuc_neu_cooling_old = 0
         s% cumulative_non_nuc_neu_cooling_older = 0

         s% cumulative_sources_and_sinks = 0
         s% cumulative_sources_and_sinks_old = 0
         s% cumulative_sources_and_sinks_older = 0
         
         s% have_initial_energy_integrals = .false.

         s% num_newton_iterations = 0
         s% num_newton_iters_stage1 = 0
         s% num_newton_iters_stage2 = 0
         s% num_newton_iters_stage3 = 0
         s% num_newton_iters_stage4 = 0
         s% num_newton_iters_stage5 = 0
         s% num_backups = 0
         s% number_of_backups_in_a_row = 0
         
         s% mesh_call_number = 0
         s% hydro_call_number = 0
         s% diffusion_call_number = 0
         s% model_number = 0
         
         s% boost_mlt_alfa = 0
         
         s% k_const_mass = 1
         s% k_below_just_added = 1
         s% k_below_const_q = 1
         s% k_CpTMdot_lt_L = 1
         
         s% why_Tlim = Tlim_struc
         
         s% done_with_center_flash = .false.
         
         s% done_with_piston = .false.
         s% piston_vfinal_inward = 0
         s% piston_vfinal_inward_old = 0
         s% piston_vfinal_inward_older = 0
         s% piston_alpha = 0
         s% piston_alpha_old = 0
         s% piston_alpha_older = 0
         
         s% len_extra_iwork = 0
         s% len_extra_work = 0
         
         s% eos_handle = 0
         s% kap_handle = 0
         
         call set_starting_star_data(s, ierr)
         if (ierr /= 0) then
            write(*,*) 'alloc_star_data failed in set_starting_star_data'
            return
         end if
         
      end function alloc_star_data

      
      subroutine null_other_new_generation(id, ierr)
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         ierr = 0
      end subroutine null_other_new_generation

      
      subroutine null_other_restore_older(id)
         integer, intent(in) :: id
      end subroutine null_other_restore_older

      
      subroutine null_other_set_current_to_old(id)
         integer, intent(in) :: id
      end subroutine null_other_set_current_to_old

      
      subroutine null_how_many_extra_header_items(id, id_extra, num_extra_cols)
         integer, intent(in) :: id, id_extra
         integer, intent(out) :: num_extra_cols
         num_extra_cols = 0
      end subroutine null_how_many_extra_header_items
   
   
      subroutine null_extra_header_items_data( &
            id, id_extra, num_extra_header_cols, &
            extra_header_col_names, extra_header_col_vals, ierr)
         integer, intent(in) :: id, id_extra, num_extra_header_cols
         character (len=*), pointer :: extra_header_col_names(:)
         real(dp), pointer :: extra_header_col_vals(:)
         integer, intent(out) :: ierr
         ierr = 0
      end subroutine null_extra_header_items_data
      
      
      subroutine set_starting_star_data(s, ierr)
         use other_wind, only: null_other_wind
         use other_adjust_mdot, only: null_other_adjust_mdot
         use other_adjust_net, only: null_other_adjust_net
         use other_torque, only: default_other_torque
         use other_torque_implicit, only: default_other_torque_implicit
         use other_momentum, only: default_other_momentum
         use other_energy, only: default_other_energy
         use other_energy_implicit, only: default_other_energy_implicit
         use other_D_mix, only: null_other_D_mix
         use other_split_mix, only: null_other_split_mix
         use other_am_mixing, only: null_other_am_mixing
         use other_atm, only: null_other_atm
         use other_brunt, only: default_other_brunt
         use other_brunt_smoothing, only: null_other_brunt_smoothing
         use other_adjust_mlt_gradT_fraction, only: &
            default_other_adjust_mlt_gradT_fraction
         use other_after_set_mixing_info, only: &
            default_other_after_set_mixing_info
         use other_diffusion, only: null_other_diffusion
         use other_mlt, only: null_other_mlt
         use other_neu, only: null_other_neu
         use other_cgrav, only: default_other_cgrav
         use other_opacity_factor, only: default_other_opacity_factor
         use other_pgstar_plots, only: null_other_pgstar_plots_info
         use other_mesh_functions
         use other_eps_grav, only: null_other_eps_grav
         use other_get_pulsation_info, only: null_other_get_pulsation_info
         use other_surface_PT, only: null_other_surface_PT
         use other_write_pulsation_info, only: null_other_write_pulsation_info
         use other_history_data_initialize, only: null_other_history_data_initialize
         use other_history_data_add_model, only: null_other_history_data_add_model
         use other_photo_write, only: default_other_photo_write
         use other_photo_read, only: default_other_photo_read
         use other_eos
         use other_kap

         type (star_info), pointer :: s
         integer, intent(out) :: ierr
         
         ! note: keep the handles for eos, kap, and net
         
         ierr = 0
         
         s% model_number = 0
         s% time = 0
         s% dt = 0
         s% dt_old = 0
         s% total_num_newton_iterations = 0
         s% num_newton_iterations = 0
         s% num_retries = 0
         s% num_backups = 0
         s% mesh_call_number = 0
         s% hydro_call_number = 0
         s% diffusion_call_number = 0
         s% model_number_for_last_retry = 0
         s% dt_limit_ratio = 0
         s% Teff = -1 ! need to calculate it
         
         s% generations = 0         
         
         s% nvar_hydro = 0                 
         s% nvar_chem = 0                 
         s% nvar = 0     
                     
         s% nz = 0     
         s% net_name = ''  
         s% species = 0                  
         s% num_reactions = 0  
               
         s% nz_old = 0         
         s% nz_older = 0         
         
         s% lnPgas_flag = .false.
         s% v_flag = .false.
         s% rotation_flag = .false.
         
         s% just_wrote_terminal_header = .false.
         s% just_did_backup = .false.

         s% doing_relax = .false.
         s% use_previous_conv_vel_from_file = .false.
         s% use_avg_burn_dxdt = .false.
         s% use_avg_mix_dxdt = .false.
         
         s% mstar_dot = 0

         s% surf_lnT = 0
         s% surf_lnd = 0
         s% surf_lnPgas = 0
         s% surf_lnR = 0
         s% surf_v = 0
         s% surf_lnS = 0
         s% surf_E = 0
        
         s% termination_code = -1
         
         s% prev_create_atm_R0_div_R = -1
         
         s% screening_mode_value = -1
         
         s% dt = -1
         s% dt_next = -1
         s% dt_next_unclipped = -1
         
         s% i_lnd = 0
         s% i_lnT = 0
         s% i_lnR = 0
         s% i_lum = 0
         s% i_lnPgas = 0
         s% i_v = 0
         s% i_chem1 = 0
         
         s% i_dv_dt = 0
         s% i_dlnT_dm = 0
         s% i_dlnd_dt = 0 
         s% i_dE_dt = 0
         s% i_dlnR_dt = 0
         s% equchem1 = 0
         
         s% op_mono_nptot = 0
         s% op_mono_ipe = 0
         s% op_mono_nrad = 0 
         s% op_mono_n = 0

         s% bcyclic_sprs_shared_net_name = ''
         s% bcyclic_shared_sprs_nonzeros = 0
         s% bcyclic_nvar_hydro = 0

         s% number_of_history_columns = -1
         s% model_number_of_history_values = -1
         s% need_to_set_history_names_etc = .true.
         s% bcyclic_nvar_hydro = 0
         
         nullify(s% finish_relax_step)
         nullify(s% finished_relax)
         
         s% how_many_extra_profile_header_items => &
            null_how_many_extra_header_items
         s% data_for_extra_profile_header_items => &
            null_extra_header_items_data
         
         s% how_many_extra_history_header_items => &
            null_how_many_extra_header_items
         s% data_for_extra_history_header_items => &
            null_extra_header_items_data
         
         s% other_wind => null_other_wind
         s% other_adjust_mdot => null_other_adjust_mdot
         s% other_adjust_net => null_other_adjust_net
         s% other_split_mix => null_other_split_mix
         s% other_D_mix => null_other_D_mix
         s% other_am_mixing => null_other_am_mixing
         s% other_torque => default_other_torque
         s% other_torque_implicit => default_other_torque_implicit
         s% other_momentum => default_other_momentum
         s% other_energy => default_other_energy
         s% other_energy_implicit => default_other_energy_implicit
         s% other_cgrav => default_other_cgrav
         s% other_opacity_factor => default_other_opacity_factor
         s% other_atm => null_other_atm
         s% other_brunt => default_other_brunt
         s% other_brunt_smoothing => null_other_brunt_smoothing
         s% other_adjust_mlt_gradT_fraction => default_other_adjust_mlt_gradT_fraction
         s% other_after_set_mixing_info => default_other_after_set_mixing_info
         s% other_diffusion => null_other_diffusion
         s% other_mlt => null_other_mlt
         s% other_neu => null_other_neu
         s% other_eps_grav => null_other_eps_grav

         s% other_eosDT_get => null_other_eosDT_get
         s% other_eosDT_get_T => null_other_eosDT_get_T
         s% other_eosDT_get_Rho => null_other_eosDT_get_Rho
         
         s% other_eosPT_get => null_other_eosPT_get
         s% other_eosPT_get_T => null_other_eosPT_get_T
         s% other_eosPT_get_Pgas => null_other_eosPT_get_Pgas
         s% other_eosPT_get_Pgas_for_Rho => null_other_eosPT_get_Pgas_for_Rho

         s% other_eosDE_get => null_other_eosDE_get

         s% other_kap_get_Type1 => null_other_kap_get_Type1
         s% other_kap_get_Type2 => null_other_kap_get_Type2
         s% other_kap_get_op_mono => null_other_kap_get_op_mono

         s% other_surface_PT => null_other_surface_PT

         s% other_pgstar_plots_info => null_other_pgstar_plots_info
         s% how_many_other_mesh_fcns => null_how_many_other_mesh_fcns
         s% other_mesh_fcn_data => null_other_mesh_fcn_data         

         s% other_write_pulsation_info => null_other_write_pulsation_info         
         s% other_get_pulsation_info => null_other_get_pulsation_info

         s% other_history_data_initialize => null_other_history_data_initialize   
         s% other_history_data_add_model => null_other_history_data_add_model

         s% other_photo_write => default_other_photo_write
         s% other_photo_read => default_other_photo_read
      
         s% other_new_generation => null_other_new_generation
         s% other_restore_older => null_other_restore_older
         s% other_set_current_to_old => null_other_set_current_to_old
         
         nullify(s% how_many_extra_history_columns)
         nullify(s% data_for_extra_history_columns)
         nullify(s% how_many_extra_profile_columns)
         nullify(s% data_for_extra_profile_columns)
         nullify(s% extra_profile_col_names)
         nullify(s% extra_profile_col_vals)
         s% num_extra_profile_cols = 0
         
         s% binary_id = 0
         s% include_binary_history_in_log_file = .false.
         s% how_many_binary_history_columns => null_how_many_binary_history_columns
         s% data_for_binary_history_columns => null_data_for_binary_history_columns
         
         s% generations = 0

         s% nz = 0
         s% nz_old = 0
         s% nz_older = 0

         s% nvar_hydro = 0
         s% nvar_chem = 0
         s% nvar = 0

         s% lnPgas_flag = .false.
         s% L_flag = .true.
         s% v_flag = .false.
         s% rotation_flag = .false.
         s% have_mixing_info = .false.
         s% doing_newton_iterations = .false.

         s% prev_Lmax = 0
         s% species = 0
         s% num_reactions = 0

         s% model_number = 0
         s% model_number_old = 0
         s% model_number_older = 0

         s% mstar = 0
         s% mstar_old = 0
         s% mstar_older = 0

         s% xmstar = 0
         s% xmstar_old = 0
         s% xmstar_older = 0

         s% M_center = 0
         s% M_center_old = 0
         s% M_center_older = 0

         s% v_center = 0
         s% v_center_old = 0
         s% v_center_older = 0

         s% R_center = 0
         s% R_center_old = 0
         s% R_center_older = 0

         s% L_center = 0
         s% L_center_old = 0
         s% L_center_older = 0

         s% time = 0
         s% time_old = 0
         s% time_older = 0

         s% total_radiation = 0
         s% total_radiation_old = 0
         s% total_radiation_older = 0

         s% total_angular_momentum = 0
         s% total_angular_momentum_old = 0
         s% total_angular_momentum_older = 0
         
         s% prev_create_atm_R0_div_R = 0

         s% dt = 0
         s% dt_old = 0

         s% have_previous_rotation_info = .false.
         s% have_previous_conv_vel = .false.
         s% have_previous_D_mix = .false.
         
         s% using_free_fall_surface_PT = .false.
         
         s% net_name = ''

         s% mstar_dot = 0
         s% mstar_dot_old = 0
         s% mstar_dot_older = 0

         s% v_surf = 0
         s% v_surf_old = 0
         s% v_surf_older = 0

         s% L_nuc_burn_total = 0
         s% L_nuc_burn_total_old = 0
         s% L_nuc_burn_total_older = 0

         s% L_by_category = 0
         s% L_by_category_old = 0
         s% L_by_category_older = 0

         s% gradT_excess_alpha = 0
         s% gradT_excess_alpha_old = 0
         s% gradT_excess_alpha_older = 0

         s% dt_limit_ratio = 0
         s% dt_limit_ratio_old = 0
         s% dt_limit_ratio_older = 0

         s% L_phot = 0
         s% L_phot_old = 0
         s% L_phot_older = 0
         s% T_surf = 0
         s% P_surf = 0

         s% h1_czb_mass = 0
         s% h1_czb_mass_old = 0
         s% h1_czb_mass_older = 0
         s% h1_czb_mass_prev = 0

         s% he_core_mass = 0
         s% he_core_mass_old = 0
         s% he_core_mass_older = 0

         s% c_core_mass = 0
         s% c_core_mass_old = 0
         s% c_core_mass_older = 0

         s% Teff = 0
         s% Teff_old = 0
         s% Teff_older = 0

         s% center_eps_nuc = 0
         s% center_eps_nuc_old = 0
         s% center_eps_nuc_older = 0

         s% Lrad_div_Ledd_avg_surf = 0
         s% Lrad_div_Ledd_avg_surf_old = 0
         s% Lrad_div_Ledd_avg_surf_older = 0

         s% w_div_w_crit_avg_surf = 0
         s% w_div_w_crit_avg_surf_old = 0
         s% w_div_w_crit_avg_surf_older = 0

         s% have_done_TP = .false.
         s% TP_state = 0
         s% TP_state_old = 0
         s% TP_state_older = 0

         s% TP_count = 0
         s% TP_count_old = 0
         s% TP_count_older = 0

         s% TP_M_H_on = 0
         s% TP_M_H_on_old = 0
         s% TP_M_H_on_older = 0

         s% TP_M_H_min = 0
         s% TP_M_H_min_old = 0
         s% TP_M_H_min_older = 0

         s% n_conv_regions = 0
         s% n_conv_regions_old = 0
         s% n_conv_regions_older = 0

         s% cz_bot_mass(:) = 0
         s% cz_bot_mass_old(:) = 0
         s% cz_bot_mass_older(:) = 0

         s% cz_top_mass(:) = 0
         s% cz_top_mass_old(:) = 0
         s% cz_top_mass_older(:) = 0

         s% dt_next = 0
         s% i_lnd = 0
         s% i_lnT = 0
         s% i_lnR = 0
         s% i_E = 0 

         s% i_lum = 0
         s% i_lnPgas = 0
         s% i_v = 0

         s% i_dv_dt = 0
         s% i_dlnT_dm = 0
         s% i_dlnd_dt = 0
         s% i_dE_dt = 0
         s% i_dlnR_dt = 0

         s% model_profile_filename = ''
         s% model_controls_filename = ''
         s% model_data_filename = ''

         s% most_recent_profile_filename = ''
         s% most_recent_controls_filename = ''

         s% most_recent_model_data_filename = ''

         s% helium_ignition = .false.
         s% carbon_ignition = .false.

         s% recent_log_header = 0
         s% phase_of_evolution = 0

         s% prev_Tcntr1 = 0
         s% prev_age1 = 0
         s% prev_Tcntr2 = 0
         s% prev_age2 = 0
         s% prev_Tsurf = 0

         s% prv_log_luminosity = 0
         s% prv_log_surface_temp = 0
         s% prv_log_center_temp = 0
         s% prv_log_center_density = 0

         s% profile_age = 0
         s% post_he_age = 0
         s% prev_luminosity = 0
         s% ignition_center_xhe = 0
         s% he_luminosity_limit = 0

         s% prev_cntr_rho = 0
         s% next_cntr_rho = 0
         s% hydro_matrix_type = 0

         s% num_newton_iterations = 0
         s% num_retries = 0
         s% num_backups = 0

         s% number_of_backups_in_a_row = 0
         s% last_backup = 0

         s% timestep_hold = 0
         s% model_number_for_last_retry = 0

         s% mesh_call_number = 0
         s% hydro_call_number = 0
         s% diffusion_call_number = 0

         s% num_rotation_solver_steps = 0
         s% num_diffusion_solver_steps = 0
         s% initial_timestep = 0
         s% why_Tlim = 0
         s% result_reason = 0

         s% have_start_values = .false.

         s% need_to_update_history_now = .false.
         s% need_to_save_profiles_now = .false.
         s% save_profiles_model_priority = 0

         s% doing_flash_wind = .false.
         s% doing_rlo_wind = .false.
         s% doing_nova_wind = .false.
         s% most_recent_photo_name = ''

         s% len_extra_iwork = 0
         s% len_extra_work = 0

      end subroutine set_starting_star_data


      subroutine create_pre_ms_model(id, ierr)
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         character (len=0) :: model_dir
         call model_builder( &
            id, model_dir, do_create_pre_ms_model, &
            .false., 'restart_photo', ierr)
      end subroutine create_pre_ms_model
      

      subroutine create_initial_model(id, ierr)
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         character (len=0) :: model_dir
         call model_builder( &
            id, model_dir, do_create_initial_model, &
            .false., 'restart_photo', ierr)
      end subroutine create_initial_model
      

      subroutine load_zams_model(id, ierr)
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         call model_builder( &
            id, '', do_load_zams_model, &
            .false., 'restart_photo', ierr)
      end subroutine load_zams_model
      

      subroutine load_saved_model(id, model_fname, ierr)
         integer, intent(in) :: id
         character (len=*), intent(in) :: model_fname
         integer, intent(out) :: ierr
         integer :: l
         l = len_trim(model_fname)
         call model_builder( &
            id, model_fname, do_load_saved_model, &
            .false., 'restart_photo', ierr)
      end subroutine load_saved_model
      

      subroutine load_restart_photo(id, restart_filename, ierr)
         integer, intent(in) :: id
         character (len=*), intent(in) :: restart_filename
         integer, intent(out) :: ierr
         call model_builder( &
            id, '', do_load_zams_model, .true., restart_filename, ierr)
      end subroutine load_restart_photo


      ! for both zams and pre-main-sequence
      subroutine model_builder( &
            id, model_info, do_which, restart, restart_filename, ierr)
         use net, only: set_net, do_micro_change_net
         use alloc, only: set_var_info
         use photo_in, only: read_star_photo
         use init_model, only: get_zams_model
         use report, only: do_report
         use star_utils, only: set_phase_of_evolution, yrs_for_init_timestep
         use adjust_xyz, only: set_z, set_y
         use pre_ms_model, only: build_pre_ms_model
         use create_initial_model, only: build_initial_model
         use read_model, only: do_read_saved_model, &
            finish_load_model, set_zero_age_params
         use relax, only: do_relax_to_limit, do_relax_mass, &
            do_relax_mass_scale, do_relax_num_steps
         integer, intent(in) :: id, do_which
         character (len=*), intent(in) :: model_info, restart_filename
         logical, intent(in) :: restart
         integer, intent(out) :: ierr
         
         type (star_info), pointer :: s
         real(dp) :: initial_mass, initial_z, dlgm_per_step
         
         real(dp), parameter :: lg_max_abs_mdot = -1000 ! use default
         real(dp), parameter :: change_mass_years_for_dt = 1
         real(dp), parameter :: min_mass_for_create_pre_ms = 0.03d0
         logical :: restore_at_end
         integer :: k

         include 'formats'
         
         ierr = 0
         
         call get_star_ptr(id, s, ierr)
         if (ierr /= 0) return

         initial_mass = s% initial_mass
         initial_z = s% initial_z
         s% dt = 0
         s% termination_code = -1
         
         if (restart) then
            s% doing_first_model_of_run = .false.
            s% doing_first_model_after_restart = .true.
            call read_star_photo(s, restart_filename, ierr)            
            if (ierr /= 0) return                     
            s% M_center = s% mstar - s% xmstar
            call check_initials
            call set_net(s, s% net_name, ierr)
            if (ierr /= 0) return
            call finish_load_model(s, ierr)
            if (s% max_years_for_timestep > 0) &
               s% dt_next = min(s% dt_next, secyer*s% max_years_for_timestep)
            return
         end if

         s% doing_first_model_of_run = .true.
         s% doing_first_model_after_restart = .false.
         call set_zero_age_params(s)
         
         if (do_which == do_load_saved_model) then
            s% dt_next = -1
            call do_read_saved_model(s, model_info, ierr)
            if (ierr /= 0) then
               write(*,*) 'load failed in do_read_saved_model'
               return
            end if
            call check_initials
            if (s% dt_next < 0) s% dt_next = yrs_for_init_timestep(s)*secyer
         else
            s% net_name = s% default_net_name
            s% species = 0
            s% lnPgas_flag = .false.
            s% L_flag = .true.
            s% v_flag = .false.
            s% rotation_flag = .false.
            s% star_mass = s% initial_mass
            s% mstar = s% initial_mass*Msun
            s% M_center = s% mstar - s% xmstar
            call set_var_info(s, ierr)
            if (ierr /= 0) then
               write(*,*) 'failed in set_var_info'
               return
            end if
            select case (do_which)
               case (do_create_initial_model)
                  if (s% initial_model_change_net) then
                     call do_micro_change_net(s, s% initial_model_new_net_name, ierr)
                  else
                     call set_net(s, s% net_name, ierr)
                  end if
                  if (ierr /= 0) then
                     write(*,*) 'failed in set_net'
                     return
                  end if
                  call build_initial_model(s, ierr)
                  if (ierr /= 0) then
                     write(*,*) 'failed in build_initial_model'
                     return
                  end if
                  s% generations = 1
                  s% dt_next = 1d-5*secyer
                  !write(*,'(a)') ' done create initial model'
                  !write(*,*)
               case (do_create_pre_ms_model)
                  if (s% initial_mass < min_mass_for_create_pre_ms) then
                     write(*,*)
                     write(*,*)
                     write(*,*)
                     write(*,'(a,1x,f5.2)') 'sorry: cannot create pre-ms smaller than', &
                        min_mass_for_create_pre_ms
                     write(*,'(a)') &
                        'please create pre-ms and then relax to lower mass as a separate operation'
                     write(*,*)
                     write(*,'(a)') 'here is an example:'
                     write(*,'(a)') 'in your inlist &controls section, set initial_mass = 0.03'
                     write(*,'(a)') 'in the &star_job section, add something like these lines'
                     write(*,'(a)') '  relax_mass_scale = .true.'
                     write(*,'(a)') '  dlgm_per_step = 1d-3 ! log10(delta M/Msun/step)'
                     write(*,'(a)') '  new_mass = 2.863362d-3 ! 3 Mjupiter in Msun units'
                     write(*,'(a)') '  change_mass_years_for_dt = 1'
                     write(*,*)
                     write(*,*)
                     write(*,*)
                     ierr = -1
                     return
                  end if
                  if (s% pre_ms_change_net) then
                     call do_micro_change_net(s, s% pre_ms_new_net_name, ierr)
                  else
                     call set_net(s, s% net_name, ierr)
                  end if
                  if (ierr /= 0) then
                     write(*,*) 'failed in set_net'
                     return
                  end if
                  write(*,2) 'species', s% species
                  
                  call build_pre_ms_model(id, s, s% nvar_hydro, s% species, ierr)
                  if (ierr /= 0) then
                     write(*,*) 'failed in build_pre_ms_model'
                     return
                  end if
                  s% generations = 1
                  s% dt_next = 1d-5*secyer
                  !write(*,'(a)') ' done create pre main-sequence model'
                  !write(*,*)
               case (do_load_zams_model)
                  s% generations = 1
                  call get_zams_model(s, s% zams_filename, ierr)
                  if (ierr /= 0) then
                     write(*,*) 'failed in get_zams_model'
                     return
                  end if
                  !write(*,1) 'after get_zams_model: dt_next', s% dt_next
                  if (s% dt_next <= 0d0) then
                     s% dt_next = yrs_for_init_timestep(s)*secyer
                     !write(*,1) 'reset: dt_next', s% dt_next
                  end if
               case default
                  write(*,*) 'bad value for do_which in build_model'
                  ierr = -1
                  return
            end select
         end if

         s% extra_heat(1:s% nz) = 0

         call finish_load_model(s, ierr)
         if (ierr /= 0) then
            write(*,*) 'failed in finish_load_model'
            return
         end if
         if (s% max_years_for_timestep > 0) &
            s% dt_next = min(s% dt_next, secyer*s% max_years_for_timestep)
         call set_phase_of_evolution(s)
         
         if (do_which == do_create_pre_ms_model) then
            if (s% mstar > s% initial_mass*Msun) then ! need to reduce mass
               write(*,1) 'reduce mass to', s% initial_mass
               !call do_relax_mass_scale( &
               !   s% id, s% initial_mass, 0.005d0, 1d0, ierr)
               call do_relax_mass(s% id, s% initial_mass, lg_max_abs_mdot, ierr)
               if (ierr /= 0) then
                  write(*,*) 'failed in do_relax_mass'
                  return
               end if
            else if (s% mstar < s% initial_mass*Msun) then ! need to increase mass
               write(*,1) 'increase mass to', s% initial_mass
               call do_relax_mass_scale( &
                  s% id, s% initial_mass, 0.005d0, 1d0, ierr)
               !call do_relax_mass(s% id, s% initial_mass, lg_max_abs_mdot, ierr)
               if (ierr /= 0) then
                  write(*,*) 'failed in do_relax_mass'
                  return
               end if
            end if
            call do_relax_num_steps( &
               s% id, s% pre_ms_relax_num_steps, s% dt_next, ierr) 
            if (ierr /= 0) then
               write(*,*) 'failed in do_relax_num_steps'
               return
            end if
         else if (do_which == do_create_initial_model .and. &
                  s% initial_model_relax_num_steps > 0) then
            call do_relax_num_steps( &
               s% id, s% initial_model_relax_num_steps, s% dt_next, ierr) 
            if (ierr /= 0) then
               write(*,*) 'failed in do_relax_num_steps'
               return
            end if
         end if

         s% doing_first_model_of_run = .true.

         
         contains
         
         subroutine check_initials
            include 'formats'
            if (abs(initial_mass - s% initial_mass) > &
                  1d-3*initial_mass .and. initial_mass > 0) then
               write(*,1) "WARNING -- inlist initial_mass ignored", initial_mass
               write(*,1) "using saved initial_mass instead", s% initial_mass
               write(*,*)
            end if
            if (abs(initial_z - s% initial_z) > &
                  1d-3*initial_z .and. initial_z > 0) then
               write(*,1) "WARNING -- inlist initial_z ignored", initial_z
               write(*,1) "using saved initial_z instead", s% initial_z
               write(*,*)
            end if
         end subroutine check_initials
         
      end subroutine model_builder
      
      
      logical function doing_restart(restart_filename)
         character (len=*) :: restart_filename
         integer :: ios
         open(unit=99, file=restart_filename, action='read', status='old', iostat=ios)
         if (ios == 0) then
            doing_restart = .true.
            close(99)
         else
            doing_restart = .false.
         end if
      end function doing_restart


      subroutine do_remove_center_at_cell_k(id, k, ierr)
         integer, intent(in) :: id, k
         integer, intent(out) :: ierr
         type (star_info), pointer :: s 
         call get_star_ptr(id, s, ierr)
         if (ierr /= 0) then
            write(*,*) 'do_remove_center_at_cell_k: get_star_ptr ierr', ierr
            return
         end if
         call do_remove_inner_fraction_q(id, s% q(k), ierr)      
      end subroutine do_remove_center_at_cell_k


      subroutine do_remove_center_by_temperature(id, temperature, ierr)
         integer, intent(in) :: id
         real(dp), intent(in) :: temperature
         integer, intent(out) :: ierr
         type (star_info), pointer :: s 
         integer :: k         
         call get_star_ptr(id, s, ierr)
         if (ierr /= 0) then
            write(*,*) 'do_remove_center_by_temperature: get_star_ptr ierr', ierr
            return
         end if
         do k=1,s% nz
            if (s% T(k) >= temperature) then
               call do_remove_inner_fraction_q(id, s% q(k), ierr)  
               return
            end if
         end do
         ierr = -1            
      end subroutine do_remove_center_by_temperature


      subroutine do_remove_center_by_radius_cm(id, r, ierr)
         integer, intent(in) :: id
         real(dp), intent(in) :: r
         integer, intent(out) :: ierr
         type (star_info), pointer :: s 
         real(dp) :: q_r, rp1, r00, qp1
         integer :: k
         
         call get_star_ptr(id, s, ierr)
         if (ierr /= 0) then
            write(*,*) 'do_remove_center_by_radius_cm: get_star_ptr ierr', ierr
            return
         end if
         rp1 = s% R_center
         if (rp1 > r) then
            ierr = -1
            if (s% report_ierr) &
               write(*,*) 'error in remove center by radius: r < R_center'
            return
         end if
         if (s% r(1) <= r) then
            ierr = -1
            if (s% report_ierr) &
               write(*,*) 'error in remove center by radius: r >= R_surface'
            return
         end if
         if (rp1 == r) return
         qp1 = 0d0
         do k=s% nz, 1, -1
            r00 = s% r(k)
            if (r00 > r .and. r >= rp1) then
               q_r = qp1 + s% dq(k)* &
                  (r*r*r - rp1*rp1*rp1)/(r00*r00*r00 - rp1*rp1*rp1)
               exit
            end if
            rp1 = r00
            qp1 = s% q(k)
         end do
         call do_remove_inner_fraction_q(id, q_r, ierr)   
            
      end subroutine do_remove_center_by_radius_cm


      subroutine do_remove_center_by_mass_gm(id, m, ierr)
         integer, intent(in) :: id
         real(dp), intent(in) :: m
         integer, intent(out) :: ierr
         type (star_info), pointer :: s 
         real(dp) :: q_m
         include 'formats'
         call get_star_ptr(id, s, ierr)
         if (ierr /= 0) then
            write(*,*) 'do_remove_center_by_mass_gm: get_star_ptr ierr', ierr
            return
         end if
         q_m = (m - s% M_center)/s% xmstar
         call do_remove_inner_fraction_q(id, q_m, ierr)      
      end subroutine do_remove_center_by_mass_gm


      subroutine do_remove_inner_fraction_q(id, q, ierr)
         use read_model, only: finish_load_model
         use alloc, only: prune_star_info_arrays
         use star_utils, only: set_qs
         integer, intent(in) :: id
         real(dp), intent(in) :: q
         integer, intent(out) :: ierr
         type (star_info), pointer :: s 
         real(dp) :: old_xmstar, new_xmstar, rp1, qp1, q00, Lp1, vp1, r00, dq_frac
         integer :: k, kk, nz
         include 'formats'
         
         call get_star_ptr(id, s, ierr)
         if (ierr /= 0) then
            write(*,*) 'do_remove_inner_fraction_q: get_star_ptr ierr', ierr
            return
         end if
         if (q < 0d0 .or. q > 1d0) then
            ierr = -1
            if (s% report_ierr) &
               write(*,*) 'error in remove center: invalid location q'
            return
         end if
         
         nz = s% nz



         
         
         ! back to 6912 way to remove center
         ! test case neutron_star_envelope fails with new way.
         do k = 1, nz
            if (s% q(k) <= q) exit
         end do
         call do_remove_center(id, k, ierr)
         return
         
         
         




         k = nz
         rp1 = s% R_center
         qp1 = 0d0
         Lp1 = s% L_center
         vp1 = s% v_center
         
         ! find k st q(k) >= q > q(k+1)
         if (q > s% q(nz)) then
            do k=nz-1,1,-1
               q00 = s% q(k)
               if (q00 >= q .and. q > qp1) then
                  rp1 = s% r(k+1)
                  Lp1 = s% L(k+1)
                  vp1 = s% v(k+1)
                  exit
               end if
               qp1 = q00
            end do
         end if
         r00 = s% r(k)
         write(*,3) 'remove cells k to nz', k, nz
         
         ! set q(k) to = q
         ! and make corresponding changes to m(k), r(k), L(k), and v(k)
         ! then remove cells from k to nz
         
         dq_frac = (q - qp1)/s% dq(k)
         
         s% m(k) = s% M_center + q*s% xmstar
         s% r(k) = pow_cr(rp1*rp1*rp1 + dq_frac*(r00*r00*r00 - rp1*rp1*rp1), 1d0/3d0)
         s% L(k) = Lp1 + dq_frac*(s% L(k) - Lp1)
         if (s% v_flag) s% v(k) = s% v(k-1) ! TESTING vp1 + dq_frac*(s% v(k) - vp1)
         s% q(k) = q

         old_xmstar = s% xmstar
         s% M_center = s% m(k)
         new_xmstar = s% m(1) - s% M_center
         s% xmstar = new_xmstar
         s% R_center = s% r(k)
         s% L_center = s% L(k)
         if (s% v_flag) s% v_center = s% v(k)
         
         s% nz = k-1
         do kk=1,k-1
            s% dq(kk) = s% dm(kk)/new_xmstar
         end do
         call set_qs(s% nz, s% q, s% dq, ierr)
         if (ierr /= 0) return
         s% generations = 1 ! memory leak, but not worth worrying about for now
         write(*,1) 'old_xmstar', old_xmstar
         write(*,1) 's% xmstar', s% xmstar
         write(*,1) 'old_xmstar/s% xmstar', old_xmstar/s% xmstar
         call prune_star_info_arrays(s, ierr)
         call finish_load_model(s, ierr)
         
      end subroutine do_remove_inner_fraction_q


      subroutine do_remove_center(id, k, ierr) ! from 6912
         use read_model, only: finish_load_model
         use alloc, only: prune_star_info_arrays
         use star_utils, only: set_qs
         integer, intent(in) :: id, k
         integer, intent(out) :: ierr
         type (star_info), pointer :: s 
         real(dp) :: old_xmstar, new_xmstar
         integer :: kk
         include 'formats'
         call get_star_ptr(id, s, ierr)
         if (ierr /= 0) then
            write(*,*) 'do_remove_center: get_star_ptr ierr', ierr
            return
         end if
         if (k <= 1 .or. k > s% nz) return

         old_xmstar = s% xmstar
         s% M_center = s% m(k)
         new_xmstar = s% m(1) - s% M_center
         s% xmstar = new_xmstar
         s% R_center = s% r(k)
         s% L_center = s% L(k)
         if (s% v_flag) s% v_center = s% v(k)
         s% nz = k-1
         do kk=1,k-1
            s% dq(kk) = s% dm(kk)/new_xmstar
         end do
         call set_qs(s% nz, s% q, s% dq, ierr)
         if (ierr /= 0) return
         s% generations = 1 ! memory leak, but not worth worrying about
         write(*,1) 'old_xmstar', old_xmstar
         write(*,1) 's% xmstar', s% xmstar
         write(*,1) 'old_xmstar/s% xmstar', old_xmstar/s% xmstar
         call prune_star_info_arrays(s, ierr)
         call finish_load_model(s, ierr)
      end subroutine do_remove_center
      
      
      integer function null_how_many_binary_history_columns(binary_id)
         integer, intent(in) :: binary_id
         null_how_many_binary_history_columns = 0
      end function null_how_many_binary_history_columns
   
   
      subroutine null_data_for_binary_history_columns( &
            binary_id, n, names, vals, ierr)
         use const_def, only: dp
         integer, intent(in) :: binary_id, n
         character (len=80) :: names(n)
         real(dp) :: vals(n)
         integer, intent(out) :: ierr
         ierr = 0
      end subroutine null_data_for_binary_history_columns

         
      end module init
