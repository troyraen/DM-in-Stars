!!! MIST SETTINGS
!! comment out in test inlists (base inlist: total MIST (inlist_OG))

!!!!!!! MLT
>     mixing_length_alpha = 1.82
>     mlt_option = 'Henyey'

!!!!!!! CONVECTION
>     alpha_semiconvection = 0.1

!!!!!!! THERMOHALINE
>     thermohaline_coeff = 666.0

!!!!!!! OPACITY
>     kappa_file_prefix = 'a09'
>     kappa_lowT_prefix = 'lowT_fa05_a09p'
>     kappa_CO_prefix = 'a09_co'
>     !CO enhanced opacities
>     kap_Type2_full_off_X = 1d-3
>     kap_Type2_full_on_X = 1d-6
>


!better resolution of the Henyey hook
>     delta_lg_XH_cntr_max = -1
!pre MS T
>       pre_ms_T_c = 5d5

>
!!!!!!! L, T limits
! limit for magnitude of max change
>     delta_lgTeff_limit = 0.005
>     delta_lgTeff_hard_limit = 0.01
>     delta_lgL_limit = 0.02
>     delta_lgL_hard_limit = 0.05

>
>
>
!!!!!!! OTHER
!CORE MASS DEFINITION
>
>     he_core_boundary_h1_fraction = 1d-4
>     c_core_boundary_he4_fraction = 1d-4
>     o_core_boundary_c12_fraction = 1d-4
>
! RATES
>     !use jina
>     set_rates_preference = .true.
>     new_rates_preference = 2

! WIND
>   use_other_wind = .true.
>   Reimers_scaling_factor = 0.1
>   Blocker_scaling_factor = 0.2
>   max_wind = 1d-3

> !MESH AND TIMESTEP PARAMETERS
>     varcontrol_target = 1d-4
>     mesh_delta_coeff = 1.0
>     max_allowed_nz = 50000

>   !to help with breathing pulses
>   include_dmu_dt_in_eps_grav = .true.

>     !to help with convergence
>     Pextra_factor = 2.0

! OVERSHOOT
>     !H core overshoot calibrated to M67
>     overshoot_f_above_nonburn_core = 0.016
>     overshoot_f_above_burn_h_core  = 0.016
>     overshoot_f_above_burn_he_core = 0.016
>     overshoot_f_above_burn_z_core  = 0.016
>     overshoot_f0_above_nonburn_core = 0.008
>     overshoot_f0_above_burn_h_core  = 0.008
>     overshoot_f0_above_burn_he_core = 0.008
>     overshoot_f0_above_burn_z_core  = 0.008
>
>     !envelope overshoot calibrated to the sun
>     overshoot_f_above_nonburn_shell = 0.0174
>     overshoot_f_below_nonburn_shell = 0.0174
>     overshoot_f_above_burn_h_shell  = 0.0174
>     overshoot_f_below_burn_h_shell  = 0.0174
>     overshoot_f_above_burn_he_shell = 0.0174
>     overshoot_f_below_burn_he_shell = 0.0174
>     overshoot_f_above_burn_z_shell  = 0.0174
>     overshoot_f_below_burn_z_shell  = 0.0174
>     overshoot_f0_above_nonburn_shell = 0.0087
>     overshoot_f0_below_nonburn_shell = 0.0087
>     overshoot_f0_above_burn_h_shell  = 0.0087
>     overshoot_f0_below_burn_h_shell  = 0.0087
>     overshoot_f0_above_burn_he_shell = 0.0087
>     overshoot_f0_below_burn_he_shell = 0.0087
>     overshoot_f0_below_burn_z_shell  = 0.0087
>     overshoot_f0_above_burn_z_shell  = 0.0087
>
>     !enhance `overshoot_f_below_nonburn_shell` by this factor during 3DUP in TPAGB
>     overshoot_below_noburn_shell_factor = 10
>
>     !multiply mesh_delta_coeff in overshooting regions by this factor
>     xtra_coef_os_above_nonburn = 0.5
>     xtra_coef_os_below_nonburn = 0.5
>     xtra_coef_os_above_burn_h = 0.5
>     xtra_coef_os_below_burn_h = 0.5
>     xtra_coef_os_above_burn_he = 0.5
>     xtra_coef_os_below_burn_he = 0.5
>     xtra_coef_os_above_burn_z = 0.5
>     xtra_coef_os_below_burn_z = 0.5
>
