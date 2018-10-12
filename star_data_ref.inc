

! Q: A lot of names are "log" something -- natural or base 10 log?
! A: I use "ln" for natural log and "log" for base 10.


! Q: How is the "center" of a zone defined? is it the center by mass, or radius?
! A: Extensive variables (R, L, M, velocity) are defined at cell boundaries.
!    Intensive variables (T, rho, composition) are cell averages.
!    We use the cell average as an estimate of the point value at the cell center by mass.
!   The themo variables are cell averages since they are calculated as functions of the cell averages.
!   Note however that gradr and gradT are not thermo variables.
!   They are defined at cell boundaries and depend on both neighbors.
!   If you are comparing to results from a model that doesn't have staggered variables
!   (i.e., one that defines the intensive and extensive vars at the same place), then
!   you will need to adjust.   For example, to compare our sound speed profile to
!   published results, we interpolate the cell centered csound values to find values
!   at cell boundaries.


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


   ! general parameters for the evolution

      ! lnPgas_flag is true if we are using lnPgas variables in place of lnd
      logical :: lnPgas_flag

      ! v_flag is true if we are using velocity variables at cell boundaries
      logical :: v_flag

      ! u_flag is true if we are using velocity variables at cell centers
      logical :: u_flag

      ! rotation_flag is true if we are using rotation variables
      logical :: rotation_flag

      ! RTI_flag is true if we are doing Rayleigh-Taylor instabilities
      logical :: RTI_flag

      ! D_omega_flag is true if we are using D_omega variables
      logical :: D_omega_flag

      ! am_nu_rot_flag is true if we are using smoothed am_nu_rot variables
      logical :: am_nu_rot_flag


      integer :: nvar_hydro ! the number of structure variables per zone

      integer :: species ! species = number of isotopes in the current net

      integer :: nvar_chem ! the number of chemical abundance variables per zone
      ! = 0 for operator splitting with separate evolution of abundances
      ! = species to solve for all abundances along with structure

      integer :: nvar ! = nvar_hydro + nvar_chem

      ! net_name identifies which net we are currently using
      character (len=net_name_len) :: net_name

      ! num_reactions = number of reactions in the current net
      integer :: num_reactions

      ! chem info
      integer, pointer :: chem_id(:) ! maps species to chem id
         ! index from 1 to species in current net
         ! value is between 1 and num_chem_isos
      integer, pointer :: net_iso(:) ! maps chem id to species number
         ! index from 1 to num_chem_isos
         ! value is 0 if the iso is not in the current net
         ! else is value between 1 and number of species in current net

      ! center boundary

      ! typically, mesa/star models an entire star and the center boundary has M, R, v, and L all = 0.
      ! however, in some situations it is useful to model only the envelope
      ! and use a nonzero center boundary.
      real(dp) :: M_center ! total baryonic mass (g) interior to innermost cell (usually 0)
      real(dp) :: v_center ! velocity (cm/s) of inner edge of innermost cell (usually 0)
      real(dp) :: R_center ! radius (cm) of inner edge of innermost cell (usually 0)
         ! R_center is automatically updated at each timestep if velocity_center is nonzero.
      real(dp) :: L_center ! luminosity (erg/s) at inner edge of innermost cell (usually 0)

      ! one reason for using a nonzero center boundary is to resolve small features in the envelope.
      ! the full model (with 0 center boundary) includes the envelope of course, but because of the
      ! limits of real(dp) we are limited to resolving envelope cell masses of about 10^-14*mstar.
      ! If that is adequate for your problem, then there is no reason to use a nonzero center boundary.
      ! But if you need to resolve smaller cell masses and you can ignore the center details,
      ! then a nonzero center can be the answer.  The code is written so that it can resolve cell masses
      ! down to about 10^-14*xmstar, where xmstar is the mass of the envelope exterior to M_center.
      ! So, for example, if xmstar = 10^-16*mstar,
      ! you can now resolve down to 10^-30*mstar in the envelope.
      ! The nonzero center boundaries are a big help with mass resolution,
      ! but they do not improve radial resolution -- that is still limited to about 10^-14*R.
      ! The remeshing algorithm knows that it must not split cells that are near that limit.
      ! In practice, this limit on radial resolution has not been a issue.
      ! For example, with R of 10 km for a neutron star, the ability to resolve 10^-8 cm is quite enough!
      ! However for mass resolution with a 1.6 Msun neutron star, we need to resolve small mass cells
      ! down to 10^10 g or less while the smallest amount of the total mass we can resolve is ~ 10^19 g.
      ! So for the NS envelope case, we need to use a nonzero center boundary.

      ! index definitions for the variables (= 0 if variable not in use)
         integer :: i_lnd ! ln(cell density average by mass)
         integer :: i_lnT ! ln cell temperature average by mass
         integer :: i_lnR ! ln radius at outer face of cell
         integer :: i_lum ! luminosity at outer face of cell
         integer :: i_v ! Lagrangian velocity at outer face of cell
         integer :: i_u ! Lagrangian velocity at center of cell
         integer :: i_lnPgas ! ln cell gas pressure average by mass
         integer :: i_alpha_RTI ! alpha for Rayleigh-Taylor instabilities

      ! chem variables, if any, follow the structure variables
         integer :: i_chem1 ! index of the first chem variable (0 means no chem variables)
            ! if i_chem1 /= 0, then i_chem1 == nvar_hydro + 1
         integer :: i_dchem1 ! index of the first delta chem variable (0 means no delta chem variables)

      ! index definitions for the equations (= 0 if equation not in use)
         integer :: i_dlnR_dt ! dlnR/dt = v/r
         integer :: i_dv_dt ! momentum conservation using v
         integer :: i_du_dt ! momentum conservation using u
         integer :: i_dlnd_dt ! mass conservation
         integer :: i_dlnE_dt ! energy conservation
         integer :: i_equL ! luminosity
         integer :: i_dalpha_RTI_dt ! alpha for Rayleigh-Taylor instabilities


      ! chem equations follow the structure equations
         integer :: equchem1 ! index of the first chem equation (0 means no chem equations)
            ! if equchem1 /= 0, then equchem1 == nvar_hydro + 1
         integer :: equdchem1 ! index of the first delta chem equation (0 means no delta chem equations)

      ! tau surface
         real(dp) :: tau_base, tau_factor
         ! tau_base is the default optical depth at the base of the atmosphere
         ! value depends on the selected option for the atmosphere model
         ! actual optical depth at base of atmosphere is tau_factor*tau_base


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



   ! the number of models that are currently stored
      integer :: generations ! between 1 and 3
         ! 1 means have current model only
         ! 2 means have current and old
         ! 3 means have current, old, and older

   ! primary data for current, old, and older models

      integer :: model_number, model_number_old, model_number_older

      ! elapsed time (seconds)
      real(dp) :: time, time_old, time_older
      ! timesteps between generations (seconds)
      real(dp) :: dt, dt_old

      ! the number of zones in the model
      integer :: nz, nz_old, nz_older

      ! total baryonic mass (grams)
      real(dp) :: mstar, mstar_old, mstar_older
      real(dp) :: xmstar, xmstar_old, xmstar_older ! baryonic mass external to M_center
         ! thus mstar = M_center + xmstar
      real(dp) :: mstar_dot, mstar_dot_old, mstar_dot_older ! (gm/second)

      ! dq(k) is the fraction of xmstar between points k and k+1
      ! NOTE: near the surface, dq's can be extremely small
      ! the code should only use q(k)-q(k+1) in cases where 0 is an okay value.
      real(dp), pointer, dimension(:) :: dq, dq_old, dq_older
      ! q is defined in terms of dq:
      !  q(1) = 1.0,
      !  for nz > k > 1, q(k) = q(k-1) - dq(k-1),
      !  q(nz) = dq(nz)
      real(dp), pointer, dimension(:) :: q, q_old, q_older

      ! rotation
      real(dp), pointer, dimension(:) :: j_rot, j_rot_old, j_rot_older
        ! j_rot(k) is specific angular momentum at outer edge of cell k; = i_rot*omega
      real(dp), pointer, dimension(:) :: omega, omega_old, omega_older
      real(dp), pointer, dimension(:) :: D_omega, D_omega_old, D_omega_older
        ! when using the implicit wind calculation, which keeps omega_div_omega_crit
        ! below a certain threshold, store the explicit value of mstar_dot
      real(dp) :: explicit_mstar_dot

      ! vectors of structure variables: xh(i,k) holds hydro structure variable i for point k.
         ! note: abundances are in xa, not in xh.
         ! also note that mass and rotation are not included in xh
      real(dp), pointer, dimension(:,:) :: xh, xh_old, xh_older ! (nvar,nz)

      ! composition variables: xa(j,k) holds mass fraction of species j for location k
      ! j is an integer from 1 to s% species
      ! chem_id(j) gives the iso identifier from chem_def
      ! net_iso(chem_id(j)) = j
      real(dp), pointer, dimension(:,:) :: xa, xa_old, xa_older

      real(dp) :: M_center_old, M_center_older
      real(dp) :: v_center_old, v_center_older
      real(dp) :: R_center_old, R_center_older
      real(dp) :: L_center_old, L_center_older


      ! structure variables
      real(dp), pointer :: m(:) ! baryonic mass coord
         ! m(k) is enclosed baryonic mass at outer edge of cell k
         ! m(k) = s% M_center + s% q(k)*s% xmstar
      real(dp), pointer :: dm(:)
         ! dm(k) is baryonic mass of cell k
         ! dm(k) = s% dq(k)*s% xmstar

      real(dp), pointer :: dm_bar(:)
         ! baryonic mass associated with cell edge
         ! for k=2,nz-1 dm_bar(k) = (dm(k-1)+dm(k))/2
         ! dm_bar(1) = dm(1)/2
         ! dm_bar(nz) = dm(nz-1)/2 + dm(nz)

      real(dp), pointer :: T(:) ! Temperature
      real(dp), pointer :: lnT(:) ! log(T)
      real(dp), pointer :: rho(:) ! density
      real(dp), pointer :: lnd(:) ! log(rho)
      real(dp), pointer :: Pgas(:) ! gas pressure (Pgas = Pion + Pelec)
      real(dp), pointer :: lnPgas(:) ! log(Pgas)
      real(dp), pointer :: L(:)
         ! L(k) is luminosity at outer edge of cell k
      real(dp), pointer :: r(:)
         ! r(k) is radius at outer edge of cell k
      real(dp), pointer :: lnR(:)
         ! lnR(k) is log(r(k))
      real(dp), pointer :: v(:) ! only used when v_flag is true
         ! v(k) is velocity at outer edge of cell k
      real(dp), pointer :: u(:) ! only used when u_flag is true
         ! u(k) is velocity at center of cell k
      real(dp), pointer :: alpha_RTI(:) ! only used when RTI_flag is true
         ! alpha_RTI(k) is measure of Rayleigh-Taylor instability at center of cell k

      real(dp), pointer :: lnE(:) ! log(specific internal energy)
      real(dp), pointer :: energy(:) ! specific internal energy (erg/g)

      ! composition information -- cell average values

      real(dp), pointer :: X(:) ! mass fraction hydrogen
      real(dp), pointer :: Y(:) ! mass fraction helium
      real(dp), pointer :: abar(:) ! average atomic weight
      real(dp), pointer :: zbar(:) ! average charge per nucleus
      real(dp), pointer :: z2bar(:) ! average charge squared per nucleus
      real(dp), pointer :: ye(:)
         ! ye is mean number free electrons per nucleon, assuming complete ionization

      ! gravitational vs baryonic mass
      real(dp), pointer :: mass_correction(:) ! = dm_gravitational/dm_baryonic
         ! calculated by chem module
      real(dp), pointer :: m_grav(:)
         ! enclosed gravitational mass at cell outer edge
         ! the variable called "m" is the enclosed baryonic mass
            ! = number of enclosed baryons * atomic mass unit
         ! NOTE: if use_mass_corrections = .false.,
         ! then m_grav = baryonic mass


      ! eos variables -- cell average values

      real(dp), pointer :: Prad(:) ! radiation pressure
      real(dp), pointer :: P(:) ! total pressure (P = Prad + Pgas)
      real(dp), pointer :: lnP(:) ! log(P)
      real(dp), pointer :: grada(:)   ! d ln T / d ln P | adiabatic
      real(dp), pointer :: dE_dRho(:) ! dE_dRho at constant T&abundances
      real(dp), pointer :: Cv(:)         ! dE/dT at constant density&abundances
      real(dp), pointer :: Cp(:)         ! heat capacity at constant pressure&abundances
         ! Cp := dh_dT at constant P, specific heat at constant pressure
         ! where h is enthalpy, h = E + P/Rho
      real(dp), pointer :: lnS(:)   ! log(specific entropy)
      real(dp), pointer :: gamma1(:)      ! adiabatic coefficient.
         ! (d_ln(p)/d_ln(rho)) at constant entropy.
      real(dp), pointer :: gamma3(:)      ! gamma3 := 1 + dlnT_dlnRho at constant S,
      real(dp), pointer :: eta(:)        ! electron degeneracy parameter
      real(dp), pointer :: theta_e(:)    ! electron degeneracy factor for graboske screening
      real(dp), pointer :: gam(:)        ! plasma interaction parameter
      real(dp), pointer :: mu(:)
         ! mean molecular weight per particle (ions + free electrons)
      real(dp), pointer :: lnfree_e(:) ! log(number of free electrons per nucleon)
      real(dp), pointer :: chiRho(:) ! dlnP_dlnRho at constant T
      real(dp), pointer :: chiT(:) ! dlnP_dlnT at constant Rho
      real(dp), pointer :: d_eos_dlnd(:,:) ! (num_eos_basic_results,nz)
      real(dp), pointer :: d_eos_dlnT(:,:) ! (num_eos_basic_results,nz)
      real(dp), pointer :: d_eos_dabar(:,:) ! (num_eos_basic_results,nz)
      real(dp), pointer :: d_eos_dzbar(:,:) ! (num_eos_basic_results,nz)
      real(dp), pointer :: dlnRho_dlnPgas_const_T(:)
      real(dp), pointer :: dlnRho_dlnT_const_Pgas(:)

      real(dp), pointer :: csound(:) ! at cell center -- set when call eos
      real(dp), pointer :: csound_face(:) ! interpolated
      real(dp), pointer :: rho_face(:) ! interpolated rho at cell outer boundary
      real(dp), pointer :: scale_height(:) ! cm
      real(dp), pointer :: v_div_csound(:) ! velocity/csound_face
      real(dp), pointer :: entropy(:) ! NOTE: kerg/baryon units
      real(dp), pointer :: grav(:) ! uses gravitational mass
      real(dp), pointer :: tau(:) ! optical depth at cell outer boundary
      real(dp), pointer :: dr_div_csound(:) ! sound crossing time for cell

      real(dp), pointer :: d_rhoface_dlnd00(:)
      real(dp), pointer :: d_rhoface_dlndm1(:)

      real(dp), pointer :: lnR_residual(:)
      real(dp), pointer :: lnd_residual(:)
      real(dp), pointer :: equL_residual(:)
      real(dp), pointer :: lnE_residual(:)
      real(dp), pointer :: E_residual(:)
      real(dp), pointer :: v_residual(:)
      real(dp), pointer :: u_residual(:)
      real(dp), pointer :: max_abs_xa_residual(:)

      real(dp), pointer :: flux_limit_R(:)
      real(dp), pointer :: erad_error(:)
      real(dp), pointer :: energy_error(:)

      real(dp), pointer :: porosity_factor(:)
      real(dp), pointer :: d_porosityF_dT(:)
      real(dp), pointer :: d_porosityF_dL(:)

      real(dp), pointer :: rmid(:) ! radius at center by mass (cm)
      real(dp), pointer :: drmid_dlnR00(:) ! partial of rmid(k) wrt lnR(k)
      real(dp), pointer :: drmid_dlnRp1(:) ! partial of rmid(k) wrt lnR(k+1)
      real(dp), pointer :: drmid2_dlnR00(:) ! partial of rmid(k)^2 wrt lnR(k)
      real(dp), pointer :: drmid2_dlnRp1(:) ! partial of rmid(k)^2 wrt lnR(k+1)

      ! rotation variables only calculated if rotatation_flag is true
         ! angular velocity at cell boundary
      real(dp), pointer :: i_rot(:) ! specific moment of interia at cell boundary
      real(dp), pointer :: fp_rot(:) ! rotation factor for momentum
      real(dp), pointer :: ft_rot(:) ! rotation factor for temperature
      real(dp), pointer :: w_div_w_crit_roche(:) ! fraction of critical rotation at the equator,
                                                 ! without the contribution of the Eddington factor
      real(dp), pointer :: r_polar(:) ! radius in polar direction
      real(dp), pointer :: r_equatorial(:) ! radius in equatorial direction
      real(dp) :: rotational_mdot_boost

      real(dp), pointer, dimension(:) :: am_nu_rot, am_nu_rot_old, am_nu_rot_older
      real(dp), pointer :: am_nu_non_rot(:) ! diffusion driven by other sources, e.g. convection

      real(dp), pointer :: am_nu_omega(:) ! for diffusion of rotation angular velocity omega
      real(dp), pointer :: am_nu_j(:) ! for diffusion of specific angular momentum j

      ! sig = (4 pi r^2 rho)^2 nu / dm
      real(dp), pointer :: am_sig_omega(:)
      real(dp), pointer :: am_sig_j(:)

      real(dp), pointer :: domega_dlnR(:) ! defined at cell boundaries
      real(dp), pointer :: richardson_number(:) ! defined at cell boundaries

      ! diffusion coefs defined at cell boundaries (cm^2/s)

      real(dp), pointer :: D_mix_non_rotation(:) ! from non-rotation sources
         ! convection, semiconvection, thermohaline, overshooting

      real(dp), pointer :: D_visc(:) ! kinematic shear viscosity
      real(dp), pointer :: D_DSI(:) ! dynamical shear instability
      real(dp), pointer :: D_SH(:) ! Solberg-Hoiland instability
      real(dp), pointer :: D_SSI(:) ! secular shear instability
      real(dp), pointer :: D_ES(:) ! Eddington-Sweet circulation
      real(dp), pointer :: D_GSF(:) ! Goldreich-Schubert-Fricke instability

      real(dp), pointer :: D_ST(:) ! Spruit dynamo mixing diffusivity
      real(dp), pointer :: nu_ST(:) ! Spruit dynamo effective viscosity
      real(dp), pointer :: omega_shear(:) ! max(1d-30,min(1d30,abs(domega_dlnr(k)/omega(k))))

      real(dp), pointer :: dynamo_B_r(:) ! magnetic field (Gauss)
      real(dp), pointer :: dynamo_B_phi(:) ! magnetic field (Gauss)


      ! kap
      real(dp), pointer :: opacity(:) ! cell average
      real(dp), pointer :: d_opacity_dlnd(:)
         ! this is partial of opacity wrt natural log of density at constant T
      real(dp), pointer :: d_opacity_dlnT(:)
         ! this is partial of opacity wrt natural log of T at constant density
      real(dp), pointer :: kap_frac_Type2(:) ! fraction of opacity from Type2 tables
      real, pointer, dimension(:) :: & ! working storage
         op_mono_umesh1, op_mono_ff1, op_mono_rs1, op_mono_s1
      integer :: op_mono_nptot, op_mono_ipe, op_mono_nrad, op_mono_n
      real(dp), pointer :: op_mono_factors(:) ! (species)
      procedure (), pointer, nopass :: set_op_mono_factors => null()
         ! called whenever the net changes

      ! net
      integer :: screening_mode_value ! values defined in screen_def
         ! screening_mode matches the screening_mode control parameter
      real(dp), pointer :: rate_factors(:) ! (num_reactions)
         ! when reaction rates are calculated, they are multiplied by the
         ! corresponding values in this array.
      procedure (), pointer, nopass :: set_rate_factors => null()
         ! called whenever the net changes

      real(dp), pointer :: category_factors(:) ! (num_categories)
         ! when reaction rates are calculated, they are multiplied by the
         ! corresponding values in this array.

      integer, pointer :: which_rates(:) ! (rates_reaction_id_max)  controls for rates
      procedure (), pointer, nopass :: set_which_rates => null()

      real(dp), pointer :: eps_nuc(:) ! sum of reaction_eps_nuc for all reactions in net
         ! thermal ergs per gram per second from nuclear reactions
         ! (nuclear reaction neutrinos have been subtracted)
      real(dp), pointer :: eps_nuc_categories(:,:) ! (num_categories, nz)
      real(dp), pointer :: d_epsnuc_dlnd(:) ! partial wrt density
      real(dp), pointer :: d_epsnuc_dlnT(:) ! partial wrt temperature
      real(dp), pointer :: d_epsnuc_dx(:,:)  ! (species,nz)
         ! d_ex_dx(j, k) is partial of eps_nuc(k) wrt species(j)

      real(dp), pointer :: eps_nuc_neu_total(:) ! erg/gm/sec as neutrinos from nuclear reactions

      real(dp), pointer :: dxdt_nuc(:,:) ! (species,nz)
         ! rate of change of mass fractions from nuclear reactions
         ! dxdt_nuc(j,k) is time derivative from nuclear burning of species(j) at point(k)
      real(dp), pointer :: dxdt_dRho(:,:) ! (species,nz)
         ! dxdt_dRho(j, k) is partial of dxdt_nuc(k) wrt density
      real(dp), pointer :: dxdt_dT(:,:) ! (species,nz)
         ! dxdt_dRho(j, k) is partial of dxdt_nuc(k) wrt temperature
      real(dp), pointer :: d_dxdt_dx(:,:,:) ! (species,species,nz)
         ! d_dxdt_dx(i,j,k) is partial of dxdt_nuc for species i
         ! wrt abundance of species j at location k

      real(dp) :: NiCo_escape_erg_s ! produced but not deposited in star.

      ! integrated eps_nuc_categories (ergs/sec)
      real(dp), pointer :: luminosity_by_category(:,:) ! (num_categories, nz)

      ! non-nuclear-reaction neutrino losses
      real(dp), pointer :: non_nuc_neu(:)
      real(dp), pointer :: d_nonnucneu_dlnd(:)
      real(dp), pointer :: d_nonnucneu_dlnT(:)

      real(dp), pointer :: nonnucneu_plas(:)
      real(dp), pointer :: nonnucneu_brem(:)
      real(dp), pointer :: nonnucneu_phot(:)
      real(dp), pointer :: nonnucneu_pair(:)
      real(dp), pointer :: nonnucneu_reco(:)


      ! irradiation heat profile
      real(dp), pointer :: irradiation_heat(:) ! (ergs/gm/sec)


      ! extra heat profile (can be set by user)
      real(dp), pointer :: extra_heat(:) ! (ergs/gm/sec)
      real(dp), pointer, dimension(:) :: &
         d_extra_heat_dlndm1, d_extra_heat_dlnd00, d_extra_heat_dlndp1, &
         d_extra_heat_dlnTm1, d_extra_heat_dlnT00, d_extra_heat_dlnTp1, &
         d_extra_heat_dlnR00, d_extra_heat_dlnRp1

      ! eps_heat = eps_nuc - non_nuc_neu + extra_heat + irradiation_heat
      real(dp), pointer :: eps_heat(:) ! (ergs/gm/sec)

      ! extra factor for opacities (can be set by user)
      real(dp), pointer, dimension(:) :: extra_opacity_factor

      ! extra dP/dm (can be set by user)
      real(dp), pointer :: extra_dPdm(:) ! at cell boundary

      ! extra torque (can be set by user)
      real(dp), pointer, dimension(:) :: &
         extra_jdot, extra_omegadot, &
         d_extra_jdot_domega_m1, d_extra_omegadot_domega_m1, &
         d_extra_jdot_domega_00, d_extra_omegadot_domega_00, &
         d_extra_jdot_domega_p1, d_extra_omegadot_domega_p1

      ! time weighted velocity
      real(dp), pointer :: vc(:) ! only used when v_flag is true
         ! if use_time_centering then
         !     vc(k) = 0.5*(v(k) + v_start(k))
         ! else
         !     vc(k) = v(k)
      real(dp) :: d_vc_dv ! 0.5 or 1.0

      ! time weighted radius^2
      real(dp), pointer :: R2(:)
         ! if use_time_centering then
         !     R2(k) = (r_start(k)^2 + r_start(k)*r(k) + r(k)^2)/3
         ! else
         !     R2(k) = r(k)^2
      real(dp), pointer :: d_R2_dlnR(:)

      ! artificial viscosity
         ! see Weaver, Zimmerman, and Woosley, ApJ, 225: 1021-1029, 1978.

      ! eta_visc and Qvisc
      real(dp), pointer :: eta_visc(:) ! viscosity (cell average)
      real(dp), pointer :: d_eta_visc_dlnd(:)
      real(dp), pointer :: d_eta_visc_dlnR00(:)
      real(dp), pointer :: d_eta_visc_dvel00(:)
      real(dp), pointer :: d_eta_visc_dlnRp1(:)
      real(dp), pointer :: d_eta_visc_dvelp1(:)

      real(dp), pointer :: Qvisc(:) ! viscous Q for cell
      real(dp), pointer :: dQvisc_dlnd(:)
      real(dp), pointer :: dQvisc_dlnR00(:)
      real(dp), pointer :: dQvisc_dvel00(:)
      real(dp), pointer :: dQvisc_dlnRp1(:)
      real(dp), pointer :: dQvisc_dvelp1(:)

      real(dp), pointer :: d_v_div_r_dm(:)
      real(dp), pointer :: d_v_div_r_dr(:)

      ! dvdt_visc for momentum equation
      real(dp), pointer :: dvdt_visc(:) ! at outer boundary of cell
      real(dp), pointer :: d_dvdt_visc_dlndm1(:)
      real(dp), pointer :: d_dvdt_visc_dlnRm1(:)
      real(dp), pointer :: d_dvdt_visc_dvm1(:)
      real(dp), pointer :: d_dvdt_visc_dlnd00(:)
      real(dp), pointer :: d_dvdt_visc_dlnR00(:)
      real(dp), pointer :: d_dvdt_visc_dv00(:)
      real(dp), pointer :: d_dvdt_visc_dlnRp1(:)
      real(dp), pointer :: d_dvdt_visc_dvp1(:)

      real(dp), pointer :: dP_dm_visc(:) ! at outer boundary of cell
      real(dp), pointer :: dlnP_dm_visc(:) ! at outer boundary of cell

      ! eps_visc for energy equation
      real(dp), pointer :: eps_visc(:) ! at cell center
      real(dp), pointer :: d_eps_visc_dlnd(:)
      real(dp), pointer :: d_eps_visc_dlnR00(:)
      real(dp), pointer :: d_eps_visc_dvel00(:)
      real(dp), pointer :: d_eps_visc_dlnRp1(:)
      real(dp), pointer :: d_eps_visc_dvelp1(:)

      ! drag
      real(dp), pointer :: dvdt_drag(:)
      real(dp), pointer :: eps_drag(:)

      ! gravitational constant (can be set by user and can be vary within model)
      real(dp), pointer :: cgrav(:)

      ! -T*Ds/Dt energy generation rate related to change in entropy
      integer :: k_below_Eulerian_eps_grav, k_Lagrangian_eps_grav
      real(dp), pointer :: eps_grav(:)
      ! partials
      real(dp), pointer, dimension(:) :: &
         d_eps_grav_dlndm1, d_eps_grav_dlnd00, d_eps_grav_dlndp1
      real(dp), pointer, dimension(:) :: &
         d_eps_grav_dlnTm1, d_eps_grav_dlnT00, d_eps_grav_dlnTp1
      real(dp), pointer, dimension(:) :: &
         d_eps_grav_dlnPgasm1_const_T, &
         d_eps_grav_dlnPgas00_const_T, &
         d_eps_grav_dlnPgasp1_const_T
      real(dp), pointer, dimension(:) :: &
         d_eps_grav_dlnTm1_const_Pgas, &
         d_eps_grav_dlnT00_const_Pgas, &
         d_eps_grav_dlnTp1_const_Pgas
      real(dp), pointer, dimension(:) :: &
         d_eps_grav_dlnR00, d_eps_grav_dlnRp1, &
         d_eps_grav_dL00, d_eps_grav_dLp1, &
         d_eps_grav_dv00, d_eps_grav_dvp1

      real(dp), pointer :: eps_grav_dm_term_const_q(:)
         ! Eulerian spatial derivative term, T*(ds/dm)*(dm/dt)_q
      real(dp), pointer :: eps_grav_dt_term_const_q(:)
         ! Eulerian time derivative term, -T*(ds/dt)_q

      real(dp), pointer :: eps_grav_composition_term(:)
         ! only set when include_composition_in_eps_grav is true

      ! for eps_grav lagrangian derivatives of lnS and mu
      integer :: prev_mesh_nz
      real(dp), pointer :: prev_mesh_xm(:) ! (prev_mesh_nz)

      real(dp), pointer :: prev_mesh_lnS(:) ! (prev_mesh_nz)
      logical :: have_prev_lnS
      real(dp), pointer :: prev_lnS(:) ! (nz) interpolated by mass
      real(dp), pointer :: prev_lnS_const_q(:) ! (nz) interpolated by q

      real(dp), pointer :: prev_mesh_mu(:) ! (prev_mesh_nz)
      logical :: have_prev_mu
      real(dp), pointer :: prev_mu(:) ! (nz) interpolated by mass

      real(dp) :: gradT_excess_alpha ! 1 for full efficiency boost; 0 for no boost
      real(dp) :: gradT_excess_min_beta
      real(dp) :: gradT_excess_max_lambda

      real(dp), pointer :: adjust_mlt_gradT_fraction(:)
         ! this vector defaults to -1.
         ! if f(k) k is >= 0 and <= 1, then
         ! gradT(k) from mlt is replaced by f(k)*grada_face(k) + (1-f(k))*gradr(k)

      ! mixing length alpha for MLT (can be set by user and can vary within model)
      real(dp), pointer :: alpha_mlt(:)

      ! NOTE: the following values are from the most recent call on mlt
         ! values for these at start of step have "_start" at end of name
         real(dp), pointer :: mlt_mixing_length(:)
            ! mixing_length_alpha * scale_height as computed by mlt
         integer, pointer :: mlt_mixing_type(:) ! as defined in mesa/const
            ! this is the value from mlt, before overshooting has been added.
         real(dp), pointer :: mlt_D(:)
            ! local mlt estimate for diffusion coeff. (cm^2/sec)
         real(dp), pointer :: mlt_vc(:)
            ! local mlt estimate for convection velocity
         real(dp), pointer :: mlt_Gamma(:)
            ! local mlt convection efficiency.
         real(dp), pointer :: mlt_cdc(:)

         real(dp), pointer :: L_conv(:)

      ! mlt results related to gradT are evaluated at each iteration and at end of step
         real(dp), pointer :: gradT_sub_grada(:) ! difference at cell boundary
         real(dp), pointer :: grada_face(:) ! grada evaluated at cell boundary

         real(dp), pointer :: actual_gradT(:) ! actual dlnT/dlnP (at boundary)
            ! (dT/T_face)/(dP/P_face) with face values intepolated by mass

         real(dp), pointer :: gradT(:) ! dlnT/dlnP (at boundary)
         real(dp), pointer :: d_gradT_dlnd00(:)
         real(dp), pointer :: d_gradT_dlnT00(:)
         real(dp), pointer :: d_gradT_dlndm1(:)
         real(dp), pointer :: d_gradT_dlnTm1(:)
         real(dp), pointer :: d_gradT_dlnR(:)
         real(dp), pointer :: d_gradT_dL(:)

         real(dp), pointer :: unsmoothed_gradT(:) ! dlnT/dlnP (at boundary)
         real(dp), pointer :: d_unsmoothed_gradT_dlnd00(:)
         real(dp), pointer :: d_unsmoothed_gradT_dlnT00(:)
         real(dp), pointer :: d_unsmoothed_gradT_dlndm1(:)
         real(dp), pointer :: d_unsmoothed_gradT_dlnTm1(:)
         real(dp), pointer :: d_unsmoothed_gradT_dlnR(:)
         real(dp), pointer :: d_unsmoothed_gradT_dL(:)

         real(dp), pointer :: gradr(:) ! grad_rad (at boundary)
         real(dp), pointer :: d_gradr_dlnd00(:)
         real(dp), pointer :: d_gradr_dlnT00(:)
         real(dp), pointer :: d_gradr_dlndm1(:)
         real(dp), pointer :: d_gradr_dlnTm1(:)
         real(dp), pointer :: d_gradr_dlnR(:)
         real(dp), pointer :: d_gradr_dL(:)

         real(dp), pointer :: conv_dP_term(:)
         real(dp), pointer :: d_conv_dP_term_dlnd00(:)
         real(dp), pointer :: d_conv_dP_term_dlnT00(:)
         real(dp), pointer :: d_conv_dP_term_dlndm1(:)
         real(dp), pointer :: d_conv_dP_term_dlnTm1(:)
         real(dp), pointer :: d_conv_dP_term_dlnR(:)
         real(dp), pointer :: d_conv_dP_term_dL(:)

         logical, pointer :: fixed_gradr_for_rest_of_newton_iters(:)


      ! adjusted (non-local) mixing info -- evaluated at start of step.

         real(dp), pointer :: cdc(:) ! convective diffusion coeff. (gm^2/sec)
         real(dp), pointer :: D_mix(:) ! diffusion coeff. (cm^2/sec)
         real(dp), pointer :: conv_vel(:) ! convection velocity
         integer, pointer :: mixing_type(:) ! mixing types are defined in mesa/mlt
         real(dp), pointer :: cz_bdy_dq(:) ! sub-cell location of convection boundary
            ! gives dq distance inward from outer edge of cell(k)
            ! where the convection zone boundary happens.
            ! i.e., q location of boundary is q(k) - cz_bdy_dq(k)
            ! this is where gradr == gradL by linear interpolation in the cell.
         logical, pointer :: newly_nonconvective(:)

         integer :: num_conv_boundaries ! boundaries of regions with mixing_type = convective_mixing
         ! boundaries are ordered from center to surface
         real(dp), pointer :: conv_bdy_q(:) ! (num_conv_boundaries)
            ! subcell location of boundary
         logical, pointer :: top_conv_bdy(:) ! (num_conv_boundaries)
         integer, pointer :: conv_bdy_loc(:) ! (num_conv_boundaries)
            ! if top_conv_bdy, top of region is between loc and loc+1
            ! else bottom of region is between loc and loc-1
         logical, pointer :: burn_h_conv_region(:) ! (num_conv_boundaries)
         logical, pointer :: burn_he_conv_region(:) ! (num_conv_boundaries)
         logical, pointer :: burn_z_conv_region(:) ! (num_conv_boundaries)

         integer :: num_mix_boundaries
         ! boundaries of regions where D_mix > mix_region_min_D_mix
         ! mix_region_min_D_mix is a parameter in &controls.
         ! boundaries given in order from center to surface
         ! no bottom boundary at loc=nz included if center is mixed
         ! however, do include top boundary at loc=1 if surface is mixed
         integer :: num_mix_regions ! number of mix boundaries with top_mix_bdy = true
         real(dp), pointer :: mix_bdy_q(:) ! (num_mix_boundaries)
            ! sub-cell location where estimate D_mix = mix_region_min_D_mix
         integer, pointer :: mix_bdy_loc(:) ! (num_mix_boundaries)
         logical, pointer :: top_mix_bdy(:) ! (num_conv_boundaries)
         logical, pointer :: burn_h_mix_region(:) ! (num_conv_boundaries)
         logical, pointer :: burn_he_mix_region(:) ! (num_conv_boundaries)
         logical, pointer :: burn_z_mix_region(:) ! (num_conv_boundaries)

         real(dp) :: core_overshoot_Hp, core_overshoot_r0, &
            core_overshoot_f0, core_overshoot_f, core_overshoot_hstep

         real(dp), pointer :: sig(:) ! sig(k) is mixing flow across face k in (gm sec^1)
            ! sig(k) is D_mix*(4*pi*r(k)**2*rho_face)**2/dmavg
         real(dp), pointer :: sig_raw(:) ! before applying limits


      real(dp), pointer :: grad_density(:) ! smoothed dlnRho/dlnP at cell boundary
      real(dp), pointer :: grad_temperature(:) ! smoothed dlnT/dlnP at cell boundary
      real(dp), pointer :: gradL(:) ! Ledoux gradient at cell boundary
         ! gradL = grada + brunt_B
      real(dp), pointer :: gradL_composition_term(:)
      integer, pointer :: dominant_iso_for_thermohaline(:)
         ! chem id for dominant isotope in setting chemical gradient term.
         ! i.e. iso with largest local dX*(1+Z)/A


      ! element diffusion
      real(dp), pointer :: diffusion_D_self(:,:) ! self coeff for chem diffusion. [cm^2/sec]
      real(dp), pointer :: edv(:,:) ! velocity [cm/s] (species,nz)
      real(dp), pointer :: v_rad(:,:) ! velocity from radiative levitation [cm/s] (species,nz)
      real(dp), pointer :: g_rad(:,:) ! acceleration from radiative levitation [cm/s^2] (species,nz)
      real(dp), pointer :: typical_charge(:,:) ! (species,nz)
      real(dp), pointer :: diffusion_dX(:,:) ! (species,nz)
      real(dp), pointer :: E_field(:) ! (nz)
      real(dp), pointer :: eps_Ne22_sedimentation(:) ! (nz)
      real(dp), pointer :: g_field_element_diffusion(:) ! (nz)


      ! gradients expected from structure equations
      ! to be compared to actual gradients in model
      ! note: these may not be set when you are using hydro-dynamics
      real(dp), pointer :: dL_dm_expected(:) ! expected difference across cell
      real(dp), pointer :: dlnP_dm_expected(:) ! expected difference at outer boundary of cell
      real(dp), pointer :: dlnT_dm_expected(:) ! expected difference at outer boundary of cell


      ! for Riemann solver
      real(dp), pointer, dimension(:) :: &
         u_face, &
         d_uface_dlnR, &
         d_uface_dL, &
         d_uface_du00, &
         d_uface_dum1, &
         d_uface_dlnd00, &
         d_uface_dlndm1, &
         d_uface_dlnT00, &
         d_uface_dlnTm1, &
         RTI_du_diffusion_kick
      real(dp), pointer, dimension(:) :: &
         P_face, &
         d_Pface_dlnR, &
         d_Pface_dL, &
         d_Pface_du00, &
         d_Pface_dum1, &
         d_Pface_dlnd00, &
         d_Pface_dlndm1, &
         d_Pface_dlnT00, &
         d_Pface_dlnTm1
      real(dp), pointer, dimension(:) :: &
         abs_du_div_cs, &
         abs_du_plus_cs


      ! power from nuclear reactions integrated from center out (ergs/second)
      real(dp), pointer :: L_nuc_burn(:) ! excluding photodisintegrations
      real(dp), pointer :: L_nuc_by_category(:,:)


      ! Brunt-Vaisala information
      real(dp), pointer :: brunt_N2(:) ! at outer boundary of cell
      real(dp), pointer :: brunt_N2_composition_term(:)
      real(dp), pointer :: brunt_B(:) ! (at points)
         ! this is the Brassard et al B := -(chiY/chiT)*(dlnY/dlnP)
      real(dp), pointer :: unsmoothed_brunt_B(:) ! pre-smoothing


      ! asteroseimology info
      real(dp) :: delta_Pg ! g-mode period spacing for l=1 (seconds)
         ! sqrt(2)*pi^2/(integral of brunt_N/r dr)
      real(dp) :: nu_max ! estimate from scaling relation (microHz)
         ! nu_max = nu_max_sun * M/Msun / ((R/Rsun)^3 (Teff/Teff_sun)^0.5)
         ! with nu_max_sun = 3100 microHz, Teff_sun = 5777
      real(dp) :: acoustic_cutoff ! 1e6*0.5*g*sqrt(gamma1*rho/P) at surface (micro Hz)
      ! asteroseismology
      real(dp) :: delta_nu ! in micro Hz.  = 10^6/(2*photosphere_acoustic_r)


      ! atmosphere structure information (only created on demand)
      integer :: atm_structure_num_pts
      real(dp), pointer :: atm_structure(:,:) ! (num_results_for_create_atm,atm_structure_num_pts)
         ! defined at points in atmosphere
         ! atm_structure(:,1) is base of atmosphere
         ! atm_structure(:,atm_structure_num_pts) is top of atmosphere

      ! for time derivatives

         ! for calculating Lagrangian time derivatives using differences at constant m
         ! holds values from previous model interpolated to new m(k)
         real(dp), pointer :: lnd_for_d_dt_const_m(:) ! not set when lnPgas_flag is true
         real(dp), pointer :: lnPgas_for_d_dt_const_m(:) ! only set when lnPgas_flag is true
         real(dp), pointer :: lnT_for_d_dt_const_m(:)
         real(dp), pointer :: lnR_for_d_dt_const_m(:)
         real(dp), pointer :: v_for_d_dt_const_m(:) ! only set when v_flag is true
         real(dp), pointer :: u_for_d_dt_const_m(:) ! only set when u_flag is true
         real(dp), pointer :: alpha_RTI_for_d_dt_const_m(:) ! only set when alpha_RTI_flag is true

         ! for calculating "Eulerian" time derivatives using differences at constant q
         ! holds values from previous model interpolated to new q(k)
         ! these are only used for eps_grav
         real(dp), pointer :: lnd_for_d_dt_const_q(:) ! not set when lnPgas_flag is true
         real(dp), pointer :: lnPgas_for_d_dt_const_q(:) ! only set when lnPgas_flag is true
         real(dp), pointer :: lnT_for_d_dt_const_q(:)

      ! lagrangian time derivatives

         real(dp), pointer :: dlnd_dt(:) ! only used when lnPgas_flag is false
         real(dp), pointer :: dlnPgas_dt(:) ! only used when lnPgas_flag is true
         real(dp), pointer :: dlnT_dt(:)
         real(dp), pointer :: dlnR_dt(:)
         real(dp), pointer :: dv_dt(:)
         real(dp), pointer :: du_dt(:)
         real(dp), pointer :: dalpha_RTI_dt(:)

         ! these are only used for eps_grav
         real(dp), pointer :: dlnd_dt_const_q(:) ! only used when lnPgas_flag is false
         real(dp), pointer :: dlnPgas_dt_const_q(:) ! only used when lnPgas_flag is true
         real(dp), pointer :: dlnT_dt_const_q(:)


      ! Rayleigh Taylor Instabilities  --  only used when RTI_flag is true

         real(dp), pointer :: dPdr_dRhodr_info(:) ! only used when RTI_flag is true
         real(dp), pointer :: dPdr_info(:) ! only used when RTI_flag is true
         real(dp), pointer :: dRhodr_info(:) ! only used when RTI_flag is true

         real(dp), pointer :: source_plus_alpha_RTI(:) ! only used when RTI_flag is true
         real(dp), pointer :: source_minus_alpha_RTI(:) ! only used when RTI_flag is true

         real(dp), pointer :: dudt_RTI(:) ! for cell
         real(dp), pointer :: dedt_RTI(:) ! for cell

         real(dp), pointer :: eta_RTI(:)
            ! eta_RTI(k) is diffusion coefficient at cell boundary
         real(dp), pointer :: etamid_RTI(:)
         real(dp), pointer :: boost_for_eta_RTI(:)

         real(dp), pointer :: sig_RTI(:) ! sig_RTI(k) is mixing flow across face k in (gm sec^1)
            ! sig_RTI(k) is eta_RTI*(4*pi*r(k)**2*rho_face)**2/dmavg
         real(dp), pointer :: sigmid_RTI(:) ! sigmid_RTI(k) is mixing flow at center k in (gm sec^1)
            ! sigmid_RTI(k) is etamid_RTI*(4*pi*rmid(k)**2*rho)**2/dm


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


   ! other info
      ! NOTE: much of this is computed at the end of the step.
      ! so it is available for "extras_check_model" for example,
      ! but not for hook routines called during the step.
      ! if you aren't sure, check the sources by grep'ing in star/private.


      ! mixing regions (from surface inward)
      ! a "mixing region" is a run of points with the same value of mixing_type
      ! i.e., a region of convective_mixing, overshoot_mixing,
      ! semiconvective_mixing, or thermohaline_mixing

      integer :: num_mixing_regions ! the number of different regions in the current model
         ! from 0 to max_num_mixing_regions
      integer :: mixing_region_type(max_num_mixing_regions) ! values from mlt_def
      integer :: mixing_region_top(max_num_mixing_regions) ! index of point at top of region
      integer :: mixing_region_bottom(max_num_mixing_regions) ! index of point at bottom of region
      integer :: largest_conv_mixing_region ! index into the mixing_region arrays

      ! convection regions
      integer :: n_conv_regions
      real(dp), dimension(max_num_mixing_regions) :: cz_bot_mass, cz_top_mass

      ! top & bottom q of 2 largest mixing regions (mx1 the largest; mx2 the 2nd largest)
      ! conv_mx for convective mixing region; mx for any kind of mixing
      real(dp) :: & ! in Mstar units --- i.e., q values, m/Mstar
         conv_mx1_top, conv_mx1_bot, conv_mx2_top, conv_mx2_bot, &
         mx1_top, mx1_bot, mx2_top, mx2_bot
      ! top & bottom r (in Rsun units) of 2 largest mixing regions
      real(dp) :: &
         conv_mx1_top_r, conv_mx1_bot_r, conv_mx2_top_r, conv_mx2_bot_r, &
         mx1_top_r, mx1_bot_r, mx2_top_r, mx2_bot_r

      ! burn zones
         ! up to 3 zones where eps_nuc > burn_min1 erg/g/s
         ! for each zone have 4 numbers: start1, start2, end2, end1
         ! start1 is mass of inner edge where first goes > burn_min1 (or -20 if none such)
         ! start2 is mass of inner edge where first zone reaches burn_min2 erg/g/sec (or -20 if none such)
         ! end2 is mass of outer edge where first zone drops back below burn_min2 erg/g/s
         ! end1 is mass of outer edge where first zone ends (i.e. eps_nuc < burn_min1)
         ! similar for second and third zones
      real(dp) :: burn_zone_mass(4,3)

      ! power emitted from nuclear reactions after subtracting neutrinos (Lsun units)
      real(dp), dimension(num_categories) :: center_eps_burn
      real(dp) :: L_nuc_burn_total, L_by_category(num_categories)

      real(dp) :: power_nuc_burn ! total thermal power excluding photodisintegrations
      real(dp) :: power_h_burn ! total thermal power from PP and CNO
      real(dp) :: power_he_burn ! total thermal power from triple-alpha
      real(dp) :: power_c_burn ! total thermal power from carbon burning

      real(dp) :: power_nuc_neutrinos ! power emitted in neutrinos (nuclear)
      real(dp) :: power_nonnuc_neutrinos ! power emitted in neutrinos (thermal)
      real(dp) :: power_neutrinos ! power emitted in neutrinos (nuclear + thermal)

      ! surface
      real(dp) :: L_surf ! (Lsun) -- depends on control tau_for_L_surf
         ! if tau_for_L_surf > 0, then is optical depth to set L_surf; else use L_surf = L(1)
      real(dp) :: log_L_surf ! log10(L_surf)
      real(dp) :: v_surf ! (cm/second)
      real(dp) :: Lrad_div_Ledd_avg_surf
      real(dp) :: w_div_w_crit_avg_surf
      real(dp) :: omega_avg_surf
      real(dp) :: omega_crit_avg_surf
      real(dp) :: j_rot_avg_surf
      real(dp) :: v_rot_avg_surf ! cm/s
      real(dp) :: v_crit_avg_surf
      real(dp) :: v_div_v_crit_avg_surf
      real(dp) :: opacity_avg_surf
      real(dp) :: logT_avg_surf
      real(dp) :: logRho_avg_surf
      real(dp) :: T_surf, P_surf ! results from atm
      real(dp) :: log_surface_luminosity ! log10(L_surf)
      real(dp) :: log_surface_radius ! log10(total radius in Rsun units)
      real(dp) :: log_surface_temperature ! log10(temperature)   ---- note: not necessarily Teff
      real(dp) :: log_surface_pressure ! log10(pressure)
      real(dp) :: log_surface_density ! log10(density)
      real(dp) :: log_surface_gravity ! log10(gravity)

      ! fractional abundances by mass near surface
         ! averages over a fraction = 'surface_avg_abundance_dq' of star mass
      real(dp) :: surface_h1
      real(dp) :: surface_he3
      real(dp) :: surface_he4
      real(dp) :: surface_c12
      real(dp) :: surface_n14
      real(dp) :: surface_o16
      real(dp) :: surface_ne20

      ! photosphere
      real(dp) :: L_phot ! photosphere luminosity (Lsun units)
      real(dp) :: photosphere_L ! Lsun units (= L_phot)
      real(dp) :: photosphere_r ! Rsun units
      real(dp) :: photosphere_m ! Msun units
      real(dp) :: photosphere_black_body_T ! black body with R and L at photosphere
      real(dp) :: photosphere_v ! cm/s
      real(dp) :: photosphere_csound ! cm/s
      real(dp) :: photosphere_opacity
      real(dp) :: photosphere_column_density ! g/cm^2
      real(dp) :: photosphere_acoustic_r ! sound travel time from center to photosphere (seconds)
      real(dp) :: Teff
      integer :: photosphere_cell_k

      ! center region -- averages over center_avg_value_dq
      real(dp) :: log_max_temperature ! log10(max temperature in kelvin)
      real(dp) :: log_center_temperature ! log10(center temperature in kelvin)
      real(dp) :: log_center_density ! log10(center density in g/cm^3)
      real(dp) :: log_center_pressure ! log10(center pressure in dynes/cm^2)
      real(dp) :: center_degeneracy
         ! the electron chemical potential in units of k*T
      real(dp) :: center_gamma
      real(dp) :: center_abar
      real(dp) :: center_zbar
      real(dp) :: center_mu
      real(dp) :: center_ye
      real(dp) :: center_eps_nuc, d_center_eps_nuc_dlnT, d_center_eps_nuc_dlnd

      real(dp) :: center_non_nuc_neu
      real(dp) :: center_entropy ! in kerg per baryon
      real(dp) :: fe_core_infall
      real(dp) :: non_fe_core_infall
      real(dp) :: max_infall_speed_mass ! Msun
      real(dp) :: center_omega
      real(dp) :: center_omega_div_omega_crit

      real(dp) :: center_h1  ! fractional abundance by mass
      real(dp) :: center_he3
      real(dp) :: center_he4
      real(dp) :: center_c12
      real(dp) :: center_n14
      real(dp) :: center_o16
      real(dp) :: center_ne20
      real(dp) :: center_si28

      real(dp) :: max_entropy ! in kerg per baryon

      ! total baryonic mass
      real(dp) :: star_mass ! total stellar baryonic mass (Msun)
      real(dp) :: star_mdot ! d(star_mass)/dt (Msun/year)
      real(dp) :: star_mass_h1
      real(dp) :: star_mass_he3
      real(dp) :: star_mass_he4
      real(dp) :: star_mass_c12
      real(dp) :: star_mass_n14
      real(dp) :: star_mass_o16
      real(dp) :: star_mass_ne20

      real(dp) :: mass_conv_core ! (Msun) 0 if core is non-convective
      real(dp) :: mass_semiconv_core ! (Msun)
         ! = mass_conv_core if there is no semiconvective region above convective core

      real(dp) :: mass_bdy_core_overshooting ! (Msun) 0 if no core overshooting
      real(dp) :: radius_bdy_core_overshooting ! (Rsun) 0 if no core overshooting

      ! abundance boundaries

      real(dp) :: he_core_mass ! baryonic (Msun)
      real(dp) :: he_core_radius ! Rsun
      real(dp) :: he_core_lgT
      real(dp) :: he_core_lgRho
      real(dp) :: he_core_L ! Lsun
      real(dp) :: he_core_v
      real(dp) :: he_core_omega ! (s^-1)
      real(dp) :: he_core_omega_div_omega_crit
      integer :: he_core_k ! boundary is in this cell

      real(dp) :: c_core_mass ! baryonic (Msun)
      real(dp) :: c_core_radius ! Rsun
      real(dp) :: c_core_lgT
      real(dp) :: c_core_lgRho
      real(dp) :: c_core_L ! Lsun
      real(dp) :: c_core_v
      real(dp) :: c_core_omega ! (s^-1)
      real(dp) :: c_core_omega_div_omega_crit
      integer :: c_core_k ! boundary is in this cell

      real(dp) :: o_core_mass ! baryonic (Msun)
      real(dp) :: o_core_radius ! Rsun
      real(dp) :: o_core_lgT
      real(dp) :: o_core_lgRho
      real(dp) :: o_core_L ! Lsun
      real(dp) :: o_core_v
      real(dp) :: o_core_omega ! (s^-1)
      real(dp) :: o_core_omega_div_omega_crit
      integer :: o_core_k ! boundary is in this cell

      real(dp) :: si_core_mass ! baryonic (Msun)
      real(dp) :: si_core_radius ! Rsun
      real(dp) :: si_core_lgT
      real(dp) :: si_core_lgRho
      real(dp) :: si_core_L ! Lsun
      real(dp) :: si_core_v
      real(dp) :: si_core_omega ! (s^-1)
      real(dp) :: si_core_omega_div_omega_crit
      integer :: si_core_k ! boundary is in this cell

      real(dp) :: fe_core_mass ! baryonic (Msun)
      real(dp) :: fe_core_radius ! Rsun
      real(dp) :: fe_core_lgT
      real(dp) :: fe_core_lgRho
      real(dp) :: fe_core_L ! Lsun
      real(dp) :: fe_core_v
      real(dp) :: fe_core_omega ! (s^-1)
      real(dp) :: fe_core_omega_div_omega_crit
      integer :: fe_core_k ! boundary is in this cell

      real(dp) :: neutron_rich_core_mass ! baryonic (Msun)
      real(dp) :: neutron_rich_core_radius ! Rsun
      real(dp) :: neutron_rich_core_lgT
      real(dp) :: neutron_rich_core_lgRho
      real(dp) :: neutron_rich_core_L ! Lsun
      real(dp) :: neutron_rich_core_v
      real(dp) :: neutron_rich_core_omega ! (s^-1)
      real(dp) :: neutron_rich_core_omega_div_omega_crit
      integer :: neutron_rich_core_k ! boundary is in this cell

      ! values at trace_mass_location
      real(dp) :: trace_mass_radius
      real(dp) :: trace_mass_lgT
      real(dp) :: trace_mass_lgRho
      real(dp) :: trace_mass_L
      real(dp) :: trace_mass_v
      real(dp) :: trace_mass_omega ! (s^-1)
      real(dp) :: trace_mass_omega_div_omega_crit
      real(dp) :: trace_mass_lgP
      real(dp) :: trace_mass_g
      real(dp) :: trace_mass_X
      real(dp) :: trace_mass_Y
      real(dp) :: trace_mass_edv_H
      real(dp) :: trace_mass_edv_He
      real(dp) :: trace_mass_scale_height
      real(dp) :: trace_mass_dlnX_dr
      real(dp) :: trace_mass_dlnY_dr
      real(dp) :: trace_mass_dlnRho_dr

      ! values at location of max T
      real(dp) :: max_T_lgT
      real(dp) :: max_T_mass ! baryonic (Msun)
      real(dp) :: max_T_radius ! (Rsun)
      real(dp) :: max_T_lgRho
      real(dp) :: max_T_lgP
      real(dp) :: max_T_lgP_thin_shell
      real(dp) :: max_T_shell_binding_energy
      real(dp) :: max_T_entropy ! (kerg/baryon)
      real(dp) :: max_T_L ! (Lsun)
      real(dp) :: max_T_eps_nuc ! (erg/g/s)

      ! values at location of max abs velocity
      real(dp) :: max_abs_v_velocity
      real(dp) :: max_abs_v_csound
      real(dp) :: max_abs_v_v_div_cs
      real(dp) :: max_abs_v_lgT
      real(dp) :: max_abs_v_lgRho
      real(dp) :: max_abs_v_lgP
      real(dp) :: max_abs_v_mass ! baryonic (Msun)
      real(dp) :: max_abs_v_radius ! (Rsun)
      real(dp) :: max_abs_v_L ! (Lsun)
      real(dp) :: max_abs_v_gamma1
      real(dp) :: max_abs_v_entropy
      real(dp) :: max_abs_v_eps_nuc

      ! info about outermost outward moving shock
      ! excluding locations with q > max_q_for_outer_mach1_location
      real(dp) :: shock_mass ! baryonic (Msun)
      real(dp) :: shock_q
      real(dp) :: shock_radius ! (Rsun)
      real(dp) :: shock_velocity
      real(dp) :: shock_csound
      real(dp) :: shock_lgT
      real(dp) :: shock_lgRho
      real(dp) :: shock_lgP
      real(dp) :: shock_gamma1
      real(dp) :: shock_entropy
      real(dp) :: shock_tau
      real(dp) :: shock_pre_lgRho
      integer :: shock_k

      ! info at innermost mach 1 location
      ! excluding locations with q < min_q_for_inner_mach1_location
      real(dp) :: inner_mach1_mass ! baryonic (Msun)
      real(dp) :: inner_mach1_q
      real(dp) :: inner_mach1_radius ! (Rsun)
      real(dp) :: inner_mach1_velocity
      real(dp) :: inner_mach1_csound
      real(dp) :: inner_mach1_lgT
      real(dp) :: inner_mach1_lgRho
      real(dp) :: inner_mach1_lgP
      real(dp) :: inner_mach1_gamma1
      real(dp) :: inner_mach1_entropy
      real(dp) :: inner_mach1_tau
      integer :: inner_mach1_k

      ! info at outermost mach 1 location
      ! excluding locations with q > max_q_for_outer_mach1_location
      real(dp) :: outer_mach1_mass ! baryonic (Msun)
      real(dp) :: outer_mach1_q
      real(dp) :: outer_mach1_radius ! (Rsun)
      real(dp) :: outer_mach1_velocity
      real(dp) :: outer_mach1_csound
      real(dp) :: outer_mach1_lgT
      real(dp) :: outer_mach1_lgRho
      real(dp) :: outer_mach1_lgP
      real(dp) :: outer_mach1_gamma1
      real(dp) :: outer_mach1_entropy
      real(dp) :: outer_mach1_tau
      integer :: outer_mach1_k

      real(dp) :: tau10_mass ! baryonic mass in solar units where optical depth = 10
      real(dp) :: tau10_radius ! radius in solar units where optical depth = 10
      real(dp) :: tau10_lgP ! estimate for log10(P) at tau = 10
      real(dp) :: tau10_lgT ! estimate for log10(T) at tau = 10
      real(dp) :: tau10_lgRho ! estimate for log10(density) at tau = 10
      real(dp) :: tau10_L ! estimate for L/Lsun at tau = 10

      real(dp) :: tau100_mass ! baryonic location in solar units where optical depth = 100
      real(dp) :: tau100_radius ! location in solar units where optical depth = 100
      real(dp) :: tau100_lgP ! estimates for values at tau = 100
      real(dp) :: tau100_lgT
      real(dp) :: tau100_lgRho
      real(dp) :: tau100_L

      ! timescales
      real(dp) :: dynamic_timescale
         ! dynamic timescale (seconds) -- estimated by 2*pi*sqrt(r^3/(G*m))
      real(dp) :: kh_timescale
         ! kelvin-helmholtz timescale (years) -- proportional to graviational energy divided by luminosity
      real(dp) :: nuc_timescale
         ! nuclear timescale (years) -- proportional to mass divided by luminosity

      ! info about the max burning locations for hydrogen, helium, and metals

      ! info about location where have max rate of hydrogen burning by PP and CNO
      real(dp) :: max_eps_h ! erg/g/s
      real(dp) :: max_eps_h_m ! baryonic mass coordinate at location of max burn (g)
      real(dp) :: max_eps_h_lgT ! log10 temperature at location of max burn
      real(dp) :: max_eps_h_lgRho ! log10 density at location of max burn
      real(dp) :: max_eps_h_lgR ! log10 radius at location of max burn
      real(dp) :: max_eps_h_lgP ! log10 pressure at location of max burn
      real(dp) :: max_eps_h_opacity ! opacity at location of max burn
      real(dp) :: max_eps_h_cp ! cp at location of max burn
      integer :: max_eps_h_k ! cell number

      ! info about location where have max rate of helium burning by triple alpha
      real(dp) :: max_eps_he ! erg/g/s
      real(dp) :: max_eps_he_m ! baryonic mass coordinate at location of max burn (g)
      real(dp) :: max_eps_he_lgT ! log10 temperature at location of max burn
      real(dp) :: max_eps_he_lgRho ! log10 density at location of max burn
      real(dp) :: max_eps_he_lgR ! log10 radius at location of max burn
      real(dp) :: max_eps_he_lgP ! log10 pressure at location of max burn
      real(dp) :: max_eps_he_opacity ! opacity at location of max burn
      real(dp) :: max_eps_he_cp ! cp at location of max burn
      integer :: max_eps_he_k ! cell number

      ! info about location where have max rate for burning other than h and he burning
      real(dp) :: max_eps_z ! erg/g/s
      real(dp) :: max_eps_z_m ! baryonic mass coordinate at location of max burn (g)
      real(dp) :: max_eps_z_lgT ! log10 temperature at location of max burn
      real(dp) :: max_eps_z_lgRho ! log10 density at location of max burn
      real(dp) :: max_eps_z_lgR ! log10 radius at location of max burn
      real(dp) :: max_eps_z_lgP ! log10 pressure at location of max burn
      real(dp) :: max_eps_z_opacity ! opacity at location of max burn
      real(dp) :: max_eps_z_cp ! cp at location of max burn
      integer :: max_eps_z_k ! cell number

      ! info about location where have max rate for burning of all kinds
      real(dp) :: max_eps_nuc ! erg/g/s
      real(dp) :: max_eps_nuc_m ! baryonic mass coordinate at location of max burn (g)
      real(dp) :: max_eps_nuc_lgT ! log10 temperature at location of max burn
      real(dp) :: max_eps_nuc_lgRho ! log10 density at location of max burn
      real(dp) :: max_eps_nuc_lgR ! log10 radius at location of max burn
      real(dp) :: max_eps_nuc_lgP ! log10 pressure at location of max burn
      real(dp) :: max_eps_nuc_opacity ! opacity at location of max burn
      real(dp) :: max_eps_nuc_cp ! cp at location of max burn
      integer :: max_eps_nuc_k ! cell number

      ! times
      real(dp) :: star_age ! time in years
      real(dp) :: time_step ! dt in years
      real(dp) :: dt_next
      real(dp) :: dt_start ! dt during first try for step

      ! a few things from start of step that will need later
      real(dp) :: prev_Lmax, prev_Ledd, starting_T_center, min_dr_div_cs_start, &
         surf_r_equatorial, surf_opacity, surf_csound, surf_rho

      ! for solvers

         integer, pointer :: ipar_decsol(:), hydro_iwork(:)
         real(dp), pointer :: rpar_decsol(:), hydro_work(:)

         real(dp), pointer :: xh_pre(:,:) ! (nvar_hydro,nz)
         real(dp), pointer :: xa_pre(:,:) ! (species,nz)
         real(dp), pointer :: xa_sub_xa_pre(:,:) ! (species,nz)

         real(dp), pointer :: lnS_pre(:) ! (nz)   for eps_grav dlnS/dt

         logical :: have_start_values
         real(dp), pointer :: lnd_start(:) ! (nz)
         real(dp), pointer :: lnP_start(:) ! (nz)
         real(dp), pointer :: P_start(:) ! (nz)
         real(dp), pointer :: lnPgas_start(:) ! (nz)
         real(dp), pointer :: lnT_start(:) ! (nz)
         real(dp), pointer :: lnE_start(:) ! (nz)
         real(dp), pointer :: energy_start(:) ! (nz)
         real(dp), pointer :: erad_start(:) ! (nz)
         real(dp), pointer :: lnS_start(:) ! (nz)
         real(dp), pointer :: lnR_start(:) ! (nz)
         real(dp), pointer :: v_start(:) ! (nz)
         real(dp), pointer :: u_start(:) ! (nz)
         real(dp), pointer :: uface_start(:) ! (nz)
         real(dp), pointer :: L_start(:) ! (nz)
         real(dp), pointer :: L_non_rad_start(:) ! (nz)
            ! L_non_rad_start is set only if freeze_non_radiative_luminosity is true.
         real(dp), pointer :: r_start(:) ! (nz)
         real(dp), pointer :: rmid_start(:) ! (nz)
         real(dp), pointer :: rho_start(:) ! (nz)
         real(dp), pointer :: omega_start(:) ! (nz)
         real(dp), pointer :: Z_start(:) ! (nz)
         real(dp), pointer :: ye_start(:) ! (nz)
         real(dp), pointer :: i_rot_start(:) ! (nz)
         real(dp), pointer :: P_div_rho_start(:)
         real(dp), pointer :: mass_correction_start(:)
         real(dp), pointer :: dxdt_nuc_start(:,:)
         real(dp), pointer :: luminosity_by_category_start(:,:) ! (num_categories,nz)
         real(dp), pointer :: scale_height_start(:) ! (nz)
         real(dp), pointer :: gradL_start(:) ! (nz)
         real(dp), pointer :: grada_start(:) ! (nz)
         real(dp), pointer :: gradr_start(:) ! (nz)
         real(dp), pointer :: grada_face_start(:) ! (nz)
         real(dp), pointer :: chiT_start(:) ! (nz)
         real(dp), pointer :: chiRho_start(:) ! (nz)
         real(dp), pointer :: cp_start(:) ! (nz)
         real(dp), pointer :: cv_start(:) ! (nz)
         real(dp), pointer :: dE_dRho_start(:) ! (nz)
         real(dp), pointer :: gam_start(:) ! (nz)
         real(dp), pointer :: eta_start(:) ! (nz)
         real(dp), pointer :: T_start(:) ! (nz)
         real(dp), pointer :: mu_start(:) ! (nz)
         real(dp), pointer :: abar_start(:) ! (nz)
         real(dp), pointer :: zbar_start(:) ! (nz)
         real(dp), pointer :: eps_nuc_start(:) ! (nz)
         real(dp), pointer :: non_nuc_neu_start(:) ! (nz)
         real(dp), pointer :: eta_visc_start(:) ! (nz)
         real(dp), pointer :: alpha_RTI_start(:) ! (nz)
         real(dp), pointer :: csound_start(:) ! (nz)
         real(dp), pointer :: tau_start(:) ! (nz)
         real(dp), pointer :: opacity_start(:) ! (nz)
         real(dp), pointer :: mlt_mixing_length_start(:)
         integer, pointer :: mlt_mixing_type_start(:)
         real(dp), pointer :: mlt_D_start(:)
         real(dp), pointer :: mlt_vc_start(:)
         real(dp), pointer :: mlt_Gamma_start(:)
         real(dp), pointer :: mlt_cdc_start(:)
         real(dp) :: shock_mass_start ! Msun units like shock_mass
         real(dp) :: photosphere_opacity_start

         ! for Lagrangian time derivatives where k < k_below_just_added
         real(dp) :: surf_lnT, surf_lnd, surf_lnPgas, &
            surf_lnR, surf_v, surf_lnS
         integer :: num_surf_revisions

         logical :: have_burner_storage
         integer :: burner_storage_sz_per_thread, burner_num_threads
         real(dp), pointer :: burner_storage(:)

         real(dp), pointer :: newton_dx(:,:)
         integer :: newton_iter, newton_adjust_iter

         real(dp), pointer, dimension(:) :: & ! (nz)
            max_burn_correction, avg_burn_correction
         real(dp), pointer :: xa_extra1(:,:) ! (species,nz) -- for use by op split
         real(dp), pointer :: xa_extra2(:,:) ! (species,nz) -- for use by op split

         integer :: k_below_const_q ! cell has same q coordinate if k < this
         integer :: k_const_mass ! cell has same mass coordinate if k >= this
         integer :: k_for_test_CpT_absMdot_div_L
         integer :: k_below_just_added
            ! "just added" == added during the current step
            ! i.e., largest k s.t. for all k' < k, cell k' has material just added this step.
            ! equivalently, this is the smallest k st. for all k' > k,
            !      material below k' was already in the star at the start of this step.
            ! if no mass was added, then this == 1.
         real(dp), pointer, dimension(:) :: &
            xh_compare_1, &  ! (nvar*nz)
            xa_compare_1     ! (species*nz)
         real(dp), pointer, dimension(:,:) :: &
            xh_compare, &   ! (nvar,nz)
            xa_compare      ! (species,nz)

      ! history data

         character (len=strlen) :: &
            model_profile_filename, model_controls_filename, model_data_filename
         character (len=strlen) :: &
            most_recent_profile_filename, most_recent_controls_filename, &
            most_recent_model_data_filename

         logical :: helium_ignition
         logical :: carbon_ignition
         integer :: recent_log_header
         integer :: phase_of_evolution
         real(dp) :: prev_Tcntr1
         real(dp) :: prev_age1
         real(dp) :: prev_Tcntr2
         real(dp) :: prev_age2
         real(dp) :: prev_Tsurf
         real(dp) :: prv_log_luminosity
         real(dp) :: prv_log_surface_temp
         real(dp) :: prv_log_center_temp
         real(dp) :: prv_log_center_density
         real(dp) :: profile_age
         real(dp) :: post_he_age
         real(dp) :: prev_luminosity
         real(dp) :: ignition_center_xhe
         real(dp) :: he_luminosity_limit
         real(dp) :: prev_cntr_rho
         real(dp) :: next_cntr_rho

      ! pgstar info
         type (pgstar_hist_node), pointer :: pgstar_hist => null()

      ! hydro matrix info
         integer :: hydro_matrix_type

      ! for remesh
         type (star_info), pointer :: other_star_info
            ! holds pointers to "other" allocated arrays
            ! these are used during remesh


      ! misc other stuff

         logical :: doing_finish_load_model, doing_relax, use_previous_conv_vel_from_file
         logical :: using_gold_tolerances, need_to_reset_cumulative_energy_info

         real(dp) :: h1_czb_mass
         real(dp) :: dt_limit_ratio ! for timestep controller
         real(dp) :: total_angular_momentum
         real(dp) :: angular_momentum_removed ! total removed by mass loss in the current step
         real(dp), pointer :: xa_removed(:) ! mass fractions for removed mass (1:species)

         logical :: have_done_TP
         integer :: TP_state
         integer :: TP_count
         real(dp) :: TP_M_H_on
         real(dp) :: TP_M_H_min

         real(dp) :: force_tau_factor, force_Tsurf_factor, force_opacity_factor
         real(dp) :: explicit_hydro_prev_dt

         integer :: total_num_newton_iterations
         integer :: num_newton_iterations
         integer :: num_solves
         integer :: num_retries
         integer :: num_backups
         integer :: num_hydro_merges
         integer :: num_hydro_splits
         integer :: number_of_backups_in_a_row
         integer :: bad_max_corr_cnt
         integer :: last_backup
         integer :: timestep_hold
         integer :: model_number_for_last_jacobian
         integer :: model_number_for_last_retry
         integer :: mesh_call_number
         integer :: burn_and_mix_call_number
         integer :: hydro_call_number
         integer :: diffusion_call_number
         integer :: num_diffusion_solver_iters
         integer :: num_diffusion_solver_steps
         integer :: num_rotation_solver_steps
         real(dp) :: max_fixup_for_mix
         real(dp) :: initial_timestep
         real(dp) :: initial_L_center, initial_R_center, initial_v_center
         real(dp) :: super_eddington_wind_mdot
         logical :: just_wrote_terminal_header
         logical :: doing_first_model_of_run
         logical :: doing_first_model_after_restart
         logical :: was_in_implicit_wind_limit, &
            was_in_implicit_wind_limit_old, &
            was_in_implicit_wind_limit_older

         real(dp) :: h1_czb_mass_prev ! set in finish_step.
            ! we need this to keep track of the last value calculated by the overshoot routine
            ! from the previous step which is not the same as the first value calculated in this step.

         integer :: Tlim_dXnuc_species
         integer :: Tlim_dXnuc_cell

         integer :: Tlim_dXnuc_drop_species
         integer :: Tlim_dXnuc_drop_cell

         integer :: Tlim_dX_species
         integer :: Tlim_dX_cell

         integer :: Tlim_dt_div_min_dr_div_cs_cell

         integer :: Tlim_dX_div_X_species
         integer :: Tlim_dX_div_X_cell

         integer :: Tlim_dlgL_nuc_category
         integer :: Tlim_dlgL_nuc_cell

         integer :: why_Tlim ! for the most recent timestep
         integer :: result_reason ! code for why the result

         logical :: need_to_update_history_now
         logical :: need_to_save_profiles_now
         integer :: save_profiles_model_priority

         ! extras -- for saving extra info along with model data for restarts
         ! old and older are for backups.  old and older are same length as current.
         ! requires a procedure call to change lengths (see "star_alloc_extras" in star_lib)
         integer :: len_extra_iwork
         integer :: len_extra_work
         integer, pointer, dimension(:) :: extra_iwork, extra_iwork_old, extra_iwork_older
         real(dp), pointer, dimension(:) :: extra_work, extra_work_old, extra_work_older

         ! more extras -- for run_star_extras applications with state.
         ! if your extra state info is all held in the following variables,
         ! restarts, retries, and backups will work without doing anything special.
         integer :: &
            ixtra1, ixtra2, ixtra3, ixtra4, ixtra5, &
            ixtra6, ixtra7, ixtra8, ixtra9, ixtra10, &
            ixtra11, ixtra12, ixtra13, ixtra14, ixtra15, &
            ixtra16, ixtra17, ixtra18, ixtra19, ixtra20, &
            ixtra21, ixtra22, ixtra23, ixtra24, ixtra25, &
            ixtra26, ixtra27, ixtra28, ixtra29, ixtra30
         real(dp) :: &
            xtra1, xtra2, xtra3, xtra4, xtra5, &
            xtra6, xtra7, xtra8, xtra9, xtra10, &
            xtra11, xtra12, xtra13, xtra14, xtra15, &
            xtra16, xtra17, xtra18, xtra19, xtra20, &
            xtra21, xtra22, xtra23, xtra24, xtra25, &
            xtra26, xtra27, xtra28, xtra29, xtra30
         logical :: &
            lxtra1, lxtra2, lxtra3, lxtra4, lxtra5, &
            lxtra6, lxtra7, lxtra8, lxtra9, lxtra10, &
            lxtra11, lxtra12, lxtra13, lxtra14, lxtra15, &
            lxtra16, lxtra17, lxtra18, lxtra19, lxtra20, &
            lxtra21, lxtra22, lxtra23, lxtra24, lxtra25, &
            lxtra26, lxtra27, lxtra28, lxtra29, lxtra30

         integer :: &
            ixtra1_old, ixtra2_old, ixtra3_old, ixtra4_old, ixtra5_old, &
            ixtra6_old, ixtra7_old, ixtra8_old, ixtra9_old, ixtra10_old, &
            ixtra11_old, ixtra12_old, ixtra13_old, ixtra14_old, ixtra15_old, &
            ixtra16_old, ixtra17_old, ixtra18_old, ixtra19_old, ixtra20_old, &
            ixtra21_old, ixtra22_old, ixtra23_old, ixtra24_old, ixtra25_old, &
            ixtra26_old, ixtra27_old, ixtra28_old, ixtra29_old, ixtra30_old
         real(dp) :: &
            xtra1_old, xtra2_old, xtra3_old, xtra4_old, xtra5_old, &
            xtra6_old, xtra7_old, xtra8_old, xtra9_old, xtra10_old, &
            xtra11_old, xtra12_old, xtra13_old, xtra14_old, xtra15_old, &
            xtra16_old, xtra17_old, xtra18_old, xtra19_old, xtra20_old, &
            xtra21_old, xtra22_old, xtra23_old, xtra24_old, xtra25_old, &
            xtra26_old, xtra27_old, xtra28_old, xtra29_old, xtra30_old
         logical :: &
            lxtra1_old, lxtra2_old, lxtra3_old, lxtra4_old, lxtra5_old, &
            lxtra6_old, lxtra7_old, lxtra8_old, lxtra9_old, lxtra10_old, &
            lxtra11_old, lxtra12_old, lxtra13_old, lxtra14_old, lxtra15_old, &
            lxtra16_old, lxtra17_old, lxtra18_old, lxtra19_old, lxtra20_old, &
            lxtra21_old, lxtra22_old, lxtra23_old, lxtra24_old, lxtra25_old, &
            lxtra26_old, lxtra27_old, lxtra28_old, lxtra29_old, lxtra30_old

         integer :: &
            ixtra1_older, ixtra2_older, ixtra3_older, ixtra4_older, ixtra5_older, &
            ixtra6_older, ixtra7_older, ixtra8_older, ixtra9_older, ixtra10_older, &
            ixtra11_older, ixtra12_older, ixtra13_older, ixtra14_older, ixtra15_older, &
            ixtra16_older, ixtra17_older, ixtra18_older, ixtra19_older, ixtra20_older, &
            ixtra21_older, ixtra22_older, ixtra23_older, ixtra24_older, ixtra25_older, &
            ixtra26_older, ixtra27_older, ixtra28_older, ixtra29_older, ixtra30_older
         real(dp) :: &
            xtra1_older, xtra2_older, xtra3_older, xtra4_older, xtra5_older, &
            xtra6_older, xtra7_older, xtra8_older, xtra9_older, xtra10_older, &
            xtra11_older, xtra12_older, xtra13_older, xtra14_older, xtra15_older, &
            xtra16_older, xtra17_older, xtra18_older, xtra19_older, xtra20_older, &
            xtra21_older, xtra22_older, xtra23_older, xtra24_older, xtra25_older, &
            xtra26_older, xtra27_older, xtra28_older, xtra29_older, xtra30_older
         logical :: &
            lxtra1_older, lxtra2_older, lxtra3_older, lxtra4_older, lxtra5_older, &
            lxtra6_older, lxtra7_older, lxtra8_older, lxtra9_older, lxtra10_older, &
            lxtra11_older, lxtra12_older, lxtra13_older, lxtra14_older, lxtra15_older, &
            lxtra16_older, lxtra17_older, lxtra18_older, lxtra19_older, lxtra20_older, &
            lxtra21_older, lxtra22_older, lxtra23_older, lxtra24_older, lxtra25_older, &
            lxtra26_older, lxtra27_older, lxtra28_older, lxtra29_older, lxtra30_older


      ! some extra arrays for use in run_star_extra.f when you need an
      ! array which is guaranteed to have a length of s% nz
      ! these arrays are NOT automatically managed during restarts/backups

         integer, pointer, dimension(:) :: &
            ixtra1_array, ixtra2_array, ixtra3_array, ixtra4_array, ixtra5_array
         real(dp), pointer, dimension(:) :: &
            xtra1_array, xtra2_array, xtra3_array, xtra4_array, xtra5_array


      ! names of variables and equations
         character (len=name_len), dimension(:), pointer :: nameofvar, nameofequ ! (nvar)

      ! equation residuals
         ! equ(i,k) is residual for equation i of cell k
         real(dp), dimension(:,:), pointer :: equ ! (nvar,nz);  equ => equ1
         real(dp), dimension(:), pointer :: equ1 ! (nvar*nz); data for equ

      ! equation Jacobian
         integer :: idiag
         real(dp), dimension(:,:), pointer :: jacobian
            ! NOT ALLOCATED. pointer set by enter_setmatrix
         ! these are pointers into the jacobian when it is block triangular

         real(dp), dimension(:,:,:), pointer :: ublk, dblk, lblk ! (nvar,nvar,nz)
         ! dblk(i,j,k) = df(i,k)/dx(j,k)
         ! lblk(i,j,k) = df(i,k)/dx(j,k-1)
         ! ublk(i,j,k) = df(i,k)/dx(j,k+1)

         real(dp), pointer, dimension(:) :: AF1 ! for factored jacobian
            ! will be allocated or reallocated as necessary by newton.
            ! deallocate when free_star
         real(qp), dimension(:,:,:), pointer :: ublk_qp, dblk_qp, lblk_qp ! (nvar,nvar,nz)
         real(qp), pointer, dimension(:) :: AF1_qp, B1_qp

      ! profile info
         integer, pointer :: profile_column_spec(:)

      ! history info
         integer :: number_of_history_columns
            ! number_of_history_columns = num_standard_history_columns + num extra history columns
         real(dp), pointer :: history_values(:) ! (number_of_history_columns)
         logical, pointer :: history_value_is_integer(:) ! (number_of_history_columns)
         character (len=maxlen_history_column_name), pointer :: &
            history_names(:) ! (number_of_history_columns)
         type (integer_dict), pointer :: history_names_dict
            ! maps names to index in history_values
         integer :: model_number_of_history_values
         logical :: need_to_set_history_names_etc
         logical :: need_to_write_history

         ! the following does not include "extra" columns
         integer, pointer :: history_column_spec(:) ! (num_standard_history_columns)

      ! sizeB info
         integer :: max_zone1, max_var1, max_zone2, max_var2
         real(dp) :: abs_max_corr1, abs_max_corr2

      ! bcyclic solver
         type(ulstore), pointer :: bcyclic_odd_storage(:)
         type(ulstore_qp), pointer :: bcyclic_odd_storage_qp(:)

      ! misc
         integer :: nvar_newton_solver
         integer :: termination_code
         character (len=strlen) :: retry_message
         logical :: numerical_jacobian, doing_numerical_jacobian
         integer :: boost_mlt_alfa
         real(dp) :: save1, save2, save3, save4, save5, save6, save7
         integer :: burn_nstep_max, burn_nfcn_total
         real(dp) :: dVARdot_dVAR
         real(dp) :: surf_accel_grav_ratio
         logical :: have_mixing_info
         logical :: doing_newton_iterations
         logical :: just_did_backup
         logical :: doing_flash_wind, doing_rlo_wind, doing_nova_wind
         real(dp) :: prev_create_atm_R0_div_R
         logical :: have_previous_rotation_info
         logical :: have_j_rot
         logical :: have_previous_RTI_info
         logical :: have_previous_conv_vel
         logical :: have_previous_D_mix
         real(dp) :: dt_next_unclipped
         real(dp) :: L_for_BB_outer_BC
         integer :: k_CpTMdot_lt_L
         integer :: dX_nuc_drop_max_k, dX_nuc_drop_max_j
         real(dp) :: dX_nuc_drop_max_drop
         real(dp) :: hydro_test_partials_val, hydro_test_partials_dval_dx
         integer :: hydro_test_partials_var

         logical :: using_revised_max_yr_dt
         real(dp) :: revised_max_yr_dt
         real(dp) :: revised_max_yr_dt_old
         real(dp) :: revised_max_yr_dt_older

         logical :: astero_using_revised_max_yr_dt
         real(dp) :: astero_revised_max_yr_dt
         real(dp) :: astero_revised_max_yr_dt_old
         real(dp) :: astero_revised_max_yr_dt_older

         integer :: binary_id
         logical :: include_binary_history_in_log_file

         real(dp), pointer, dimension(:) :: prev_conv_vel_from_file

         ! variation control value for timestep controller
         real(dp) :: dt_limit_ratio_old, dt_limit_ratio_older

         ! power emitted from nuclear reactions after subtracting neutrinos (Lsun units)
         real(dp) :: L_nuc_burn_total_old, L_nuc_burn_total_older
         real(dp), dimension(num_categories) :: L_by_category_old, L_by_category_older

         ! surface info
         real(dp) :: v_surf_old, v_surf_older ! (cm/second)
         real(dp) :: L_phot_old, L_phot_older ! photosphere luminosity (Lsun units)
         real(dp) :: Lrad_div_Ledd_avg_surf_old, Lrad_div_Ledd_avg_surf_older
         real(dp) :: w_div_w_crit_avg_surf_old, w_div_w_crit_avg_surf_older

         ! convection regions
         integer :: n_conv_regions_old, n_conv_regions_older
         real(dp), dimension(max_num_mixing_regions) :: &
            cz_bot_mass_old, cz_bot_mass_older, &
            cz_top_mass_old, cz_top_mass_older

         ! thermal pulse info
         integer :: TP_state_old, TP_state_older
         integer :: TP_count_old, TP_count_older
         real(dp) :: TP_M_H_on_old, TP_M_H_on_older
         real(dp) :: TP_M_H_min_old, TP_M_H_min_older

         real(dp), pointer, dimension(:) :: lnT_old, lnT_older
         real(dp), pointer, dimension(:) :: conv_vel_old, conv_vel_older
         real(dp), pointer, dimension(:) :: nu_ST_old, nu_ST_older
         real(dp), pointer, dimension(:) :: D_ST_old, D_ST_older
         real(dp), pointer, dimension(:) :: D_DSI_old, D_DSI_older
         real(dp), pointer, dimension(:) :: D_SH_old, D_SH_older
         real(dp), pointer, dimension(:) :: D_SSI_old, D_SSI_older
         real(dp), pointer, dimension(:) :: D_ES_old, D_ES_older
         real(dp), pointer, dimension(:) :: D_GSF_old, D_GSF_older
         real(dp), pointer, dimension(:) :: D_mix_old, D_mix_older
         real(dp), pointer, dimension(:) :: dPdr_dRhodr_info_old, dPdr_dRhodr_info_older

         real(dp), pointer, dimension(:) :: max_abs_xa_corr ! from most recent newton iteration

         real(dp) :: gradT_excess_alpha_old, gradT_excess_alpha_older
         real(dp) :: total_angular_momentum_old, total_angular_momentum_older
         real(dp) :: h1_czb_mass_old, h1_czb_mass_older ! baryonic (Msun)
         real(dp) :: he_core_mass_old, he_core_mass_older
         real(dp) :: c_core_mass_old, c_core_mass_older
         real(dp) :: Teff_old, Teff_older
         real(dp) :: Lsurf_m, Lsurf_m_old, Lsurf_m_older
         real(dp) :: center_eps_nuc_old, center_eps_nuc_older

         real(dp) :: &
            total_internal_energy_start, &
            total_gravitational_energy_start, &
            total_linear_kinetic_energy_start, &
            total_rotational_kinetic_energy_start, &
            total_energy_start, &
            total_internal_energy, &
            total_gravitational_energy, &
            total_linear_kinetic_energy, &
            total_rotational_kinetic_energy, &
            total_energy, &
            total_eps_grav, &
            total_radiation, &
            acoustic_L, &
            cumulative_acoustic_L, &
            acoustic_L_center, &
            cumulative_acoustic_L_center, &
            total_nuclear_heating, &
            total_non_nuc_neu_cooling, &
            total_irradiation_heating, &
            total_Ne22_sedimentation_heating, &
            total_extra_heating, &
            total_energy_sources_and_sinks, &
            error_in_energy_conservation, &
            cumulative_eps_grav, &
            cumulative_delta_total_energy, &
            cumulative_energy_error, &
            cumulative_L_center, &
            cumulative_L_surf, &
            cumulative_extra_heating, &
            cumulative_irradiation_heating, &
            cumulative_Ne22_sedimentation_heating, &
            cumulative_nuclear_heating, &
            cumulative_non_nuc_neu_cooling, &
            cumulative_sources_and_sinks, &
            log_P_center, &
            min_kap_floor

         real(dp) :: &
            min_kap_floor_old, &
            min_kap_floor_older

         real(dp) :: &
            log_P_center_old, &
            log_P_center_older

         real(dp) :: &
            cumulative_delta_total_energy_old, &
            cumulative_delta_total_energy_older

         real(dp) :: &
            cumulative_energy_error_old, &
            cumulative_energy_error_older

         real(dp) :: &
            cumulative_acoustic_L_old, &
            cumulative_acoustic_L_older

         real(dp) :: &
            cumulative_acoustic_L_center_old, &
            cumulative_acoustic_L_center_older

         real(dp) :: &
            cumulative_eps_grav_old, &
            cumulative_eps_grav_older

         real(dp) :: &
            cumulative_L_surf_old, &
            cumulative_L_surf_older

         real(dp) :: &
            cumulative_L_center_old, &
            cumulative_L_center_older

         real(dp) :: &
            cumulative_extra_heating_old, &
            cumulative_extra_heating_older

         real(dp) :: &
            cumulative_irradiation_heating_old, &
            cumulative_irradiation_heating_older

         real(dp) :: &
            cumulative_Ne22_sedimentation_heating_old, &
            cumulative_Ne22_sedimentation_heating_older

         real(dp) :: &
            cumulative_nuclear_heating_old, &
            cumulative_nuclear_heating_older

         real(dp) :: &
            cumulative_non_nuc_neu_cooling_old, &
            cumulative_non_nuc_neu_cooling_older

         real(dp) :: &
            cumulative_sources_and_sinks_old, &
            cumulative_sources_and_sinks_older

         real(dp) :: &
            total_radiation_old, &
            total_radiation_older

         real(dp) :: mesh_adjust_IE_conservation
         real(dp) :: mesh_adjust_PE_conservation
         real(dp) :: mesh_adjust_KE_conservation

         character (len=strlen) :: most_recent_photo_name

         logical :: have_initial_energy_integrals
         real(dp) :: &
            total_internal_energy_initial, &
            total_gravitational_energy_initial, &
            total_linear_kinetic_energy_initial, &
            total_rotational_kinetic_energy_initial, &
            total_energy_initial

         logical :: using_revised_net_name
         character (len=net_name_len) :: revised_net_name
         character (len=net_name_len) :: revised_net_name_old
         character (len=net_name_len) :: revised_net_name_older

         integer :: num_extra_profile_cols
         character (len=80), pointer :: extra_profile_col_names(:)
         real(dp), pointer :: extra_profile_col_vals(:,:) ! (nz,num_extra)



      ! system clock
         integer :: starting_system_clock_time
         integer :: current_system_clock_time
         integer :: system_clock_rate
         real(dp) :: total_elapsed_time ! (current - starting)/rate


      ! timing information
         logical :: doing_timing
         integer :: system_clock_at_start_of_step

!         use star_utils, only: start_time, update_time
!         integer :: time0
!         real(dp) :: total
!         if (s% doing_timing) call start_time(s, time0, total)
!         <work>
!         if (s% doing_timing) call update_time(s, time0, total, s% time_<work>)

         ! if change this list, also must change
                  ! total_times function in star_utils
                  ! and init to set the time to 0.0 at start.
            real(dp) :: &
               time_evolve_step, &
               time_remesh, &
               time_adjust_mass, &
               time_element_diffusion, &
               time_semi_explicit_hydro, &
               time_struct_burn_mix, &
               time_newton_matrix, &
               time_solve_mix, &
               time_solve_burn, &
               time_solve_omega_mix, &
               time_eos, &
               time_neu_kap, &
               time_nonburn_net, &
               time_mlt, &
               time_set_hydro_vars
               ! change total_times function in star_utils if change this list

         real(dp) :: time_total

         procedure(id_only_function_interface), pointer, nopass :: &
            finish_relax_step => null()
         procedure(id_only_subroutine_interface), pointer, nopass :: &
            finished_relax => null()

         procedure(how_many_extra_header_items_interface), pointer, nopass :: &
            how_many_extra_profile_header_items => null()
         procedure(extra_header_items_data_interface), pointer, nopass :: &
            data_for_extra_profile_header_items => null()

         procedure(how_many_extra_header_items_interface), pointer, nopass :: &
            how_many_extra_history_header_items => null()
         procedure(extra_header_items_data_interface), pointer, nopass :: &
            data_for_extra_history_header_items => null()

      ! "extras" procedures
         procedure(extras_startup_interface), pointer, nopass :: &
            extras_startup => null()

         procedure(extras_check_model_interface), pointer, nopass :: &
            extras_check_model => null()

         procedure(extras_start_step_interface), pointer, nopass :: &
            extras_start_step => null()

         procedure(extras_finish_step_interface), pointer, nopass :: &
            extras_finish_step => null()

         procedure(extras_after_evolve_interface), pointer, nopass :: &
            extras_after_evolve => null()


      ! "other" procedures

         procedure(other_adjust_net_interface), pointer, nopass :: &
            other_adjust_net => null()
         procedure(other_adjust_mdot_interface), pointer, nopass :: &
            other_adjust_mdot => null()
         procedure(other_j_for_adjust_J_lost_interface), pointer, nopass :: &
            other_j_for_adjust_J_lost => null()
         procedure(other_after_struct_burn_mix_interface), pointer, nopass :: &
            other_after_struct_burn_mix => null()
         procedure(other_before_struct_burn_mix_interface), pointer, nopass :: &
            other_before_struct_burn_mix => null()
         procedure(other_wind_interface), pointer, nopass :: &
            other_wind => null()
         procedure (other_split_mix_interface), pointer, nopass :: &
            other_split_mix => null()
         procedure (other_D_mix_interface), pointer, nopass :: &
            other_D_mix => null()
         procedure (other_am_mixing_interface), pointer, nopass :: &
            other_am_mixing => null()
         procedure (other_momentum_interface), pointer, nopass :: &
            other_momentum => null()
         procedure (other_energy_interface), pointer, nopass :: &
            other_energy => null()
         procedure (other_paquette_coefficients_interface), pointer, nopass :: &
            other_paquette_coefficients => null()
         procedure (other_energy_implicit_interface), pointer, nopass :: &
            other_energy_implicit => null()
         procedure (other_eval_fp_ft_interface), pointer, nopass :: &
            other_eval_fp_ft => null()
         procedure (other_torque_interface), pointer, nopass :: &
            other_torque => null()
         procedure (other_torque_implicit_interface), pointer, nopass :: &
            other_torque_implicit => null()
         procedure (other_brunt_interface), pointer, nopass :: &
            other_brunt => null()
         procedure (other_brunt_smoothing_interface), pointer, nopass :: &
            other_brunt_smoothing => null()
         procedure (other_build_initial_model_interface), pointer, nopass :: &
            other_build_initial_model => null()
         procedure (other_cgrav_interface), pointer, nopass :: &
            other_cgrav => null()
         procedure (other_alpha_mlt_interface), pointer, nopass :: &
            other_alpha_mlt => null()
         procedure (other_opacity_factor_interface), pointer, nopass :: &
            other_opacity_factor => null()
         procedure (other_eps_grav_interface), pointer, nopass :: &
            other_eps_grav => null()
         procedure (other_porosity_factor_interface), pointer, nopass :: &
            other_porosity_factor => null()

         procedure (other_mlt_interface), pointer, nopass :: &
            other_mlt => null()
         procedure (other_newton_monitor_interface), pointer, nopass :: &
            other_newton_monitor => null()
         procedure (other_neu_interface), pointer, nopass :: &
            other_neu => null()
         procedure (other_net_get_interface), pointer, nopass :: &
            other_net_get => null()
         procedure (other_atm_interface), pointer, nopass :: &
            other_atm => null()
         procedure (other_adjust_mlt_gradT_fraction_interface), pointer, nopass :: &
            other_adjust_mlt_gradT_fraction => null()
         procedure (other_after_set_mixing_info_interface), pointer, nopass :: &
            other_after_set_mixing_info => null()
         procedure (other_diffusion_interface), pointer, nopass :: &
            other_diffusion => null()

         procedure (other_surface_PT_interface), pointer, nopass :: &
            other_surface_PT => null()

         procedure (other_eosDT_get_interface), pointer, nopass :: &
            other_eosDT_get => null()
         procedure (other_eosDT_get_T_interface), pointer, nopass :: &
            other_eosDT_get_T => null()
         procedure (other_eosDT_get_Rho_interface), pointer, nopass :: &
            other_eosDT_get_Rho => null()

         procedure (other_eosPT_get_interface), pointer, nopass :: &
            other_eosPT_get => null()
         procedure (other_eosPT_get_T_interface), pointer, nopass :: &
            other_eosPT_get_T => null()
         procedure (other_eosPT_get_Pgas_interface), pointer, nopass :: &
            other_eosPT_get_Pgas => null()
         procedure (other_eosPT_get_Pgas_for_Rho_interface), pointer, nopass :: &
            other_eosPT_get_Pgas_for_Rho => null()

         procedure (other_eosDE_get_interface), pointer, nopass :: &
            other_eosDE_get => null()

         procedure (other_kap_get_interface), pointer, nopass :: &
            other_kap_get => null()
         procedure (other_kap_get_op_mono_interface), pointer, nopass :: &
            other_kap_get_op_mono => null()

         procedure (other_pgstar_plots_info_interface), pointer, nopass :: &
            other_pgstar_plots_info => null()
         procedure (how_many_other_mesh_fcns_interface), pointer, nopass :: &
            how_many_other_mesh_fcns => null()
         procedure (other_mesh_fcn_data_interface), pointer, nopass :: &
            other_mesh_fcn_data => null()

         procedure (other_write_pulsation_info_interface), pointer, nopass :: &
            other_write_pulsation_info => null()
         procedure (other_get_pulsation_info_interface), pointer, nopass :: &
            other_get_pulsation_info => null()

         procedure (other_history_data_initialize_interface), pointer, nopass :: &
            other_history_data_initialize => null()
         procedure (other_history_data_add_model_interface), pointer, nopass :: &
            other_history_data_add_model => null()

         procedure (other_photo_write_interface), pointer, nopass :: &
            other_photo_write => null()
         procedure (other_photo_read_interface), pointer, nopass :: &
            other_photo_read => null()

         procedure (other_new_generation_interface), pointer, nopass :: &
            other_new_generation => null()
         procedure (other_restore_older_interface), pointer, nopass :: &
            other_restore_older => null()
         procedure (other_set_current_to_old_interface), pointer, nopass :: &
            other_set_current_to_old => null()

         procedure (other_astero_freq_corr_interface), pointer, nopass :: &
            other_astero_freq_corr => null()

         procedure (how_many_extra_history_columns_interface), pointer, nopass :: &
            how_many_extra_history_columns => null()
         procedure (data_for_extra_history_columns_interface), pointer, nopass :: &
            data_for_extra_history_columns => null()

         procedure (how_many_extra_profile_columns_interface), pointer, nopass :: &
            how_many_extra_profile_columns => null()
         procedure (data_for_extra_profile_columns_interface), pointer, nopass :: &
            data_for_extra_profile_columns => null()

         procedure (how_many_binary_history_columns_interface), pointer, nopass :: &
            how_many_binary_history_columns => null()
         procedure (data_for_binary_history_columns_interface), pointer, nopass :: &
            data_for_binary_history_columns => null()

         procedure(other_export_pulse_data_interface), pointer, nopass :: &
              other_export_pulse_data => null()
         procedure(other_get_pulse_data_interface), pointer, nopass :: &
              other_get_pulse_data => null()
         procedure (other_edit_pulse_data_interface), pointer, nopass :: &
              other_edit_pulse_data => null()

         procedure (binary_other_torque_implicit_interface), pointer, nopass :: &
            binary_other_torque_implicit => null()
         procedure (binary_other_torque_interface), pointer, nopass :: &
            binary_other_torque => null()

         procedure(pgstar_decorator_interface), pointer, nopass :: &
            Abundance_pgstar_decorator => null(), &
            Color_Magnitude1_pgstar_decorator => null(), &
            Color_Magnitude2_pgstar_decorator => null(), &
            Color_Magnitude3_pgstar_decorator => null(), &
            Color_Magnitude4_pgstar_decorator => null(), &
            Color_Magnitude5_pgstar_decorator => null(), &
            Color_Magnitude6_pgstar_decorator => null(), &
            Color_Magnitude7_pgstar_decorator => null(), &
            Color_Magnitude8_pgstar_decorator => null(), &
            Color_Magnitude9_pgstar_decorator => null(), &
            dPg_dnu_pgstar_decorator => null(), &
            Dynamo_pgstar_decorator => null(), &
            History_Panels1_pgstar_decorator => null(), &
            History_Panels2_pgstar_decorator => null(), &
            History_Panels3_pgstar_decorator => null(), &
            History_Panels4_pgstar_decorator => null(), &
            History_Panels5_pgstar_decorator => null(), &
            History_Panels6_pgstar_decorator => null(), &
            History_Panels7_pgstar_decorator => null(), &
            History_Panels8_pgstar_decorator => null(), &
            History_Panels9_pgstar_decorator => null(), &
            History_Track1_pgstar_decorator => null(), &
            History_Track2_pgstar_decorator => null(), &
            History_Track3_pgstar_decorator => null(), &
            History_Track4_pgstar_decorator => null(), &
            History_Track5_pgstar_decorator => null(), &
            History_Track6_pgstar_decorator => null(), &
            History_Track7_pgstar_decorator => null(), &
            History_Track8_pgstar_decorator => null(), &
            History_Track9_pgstar_decorator => null(), &
            HR_pgstar_decorator => null(), &
            Kipp_pgstar_decorator => null(), &
            logg_logt_pgstar_decorator => null(), &
            logg_teff_pgstar_decorator => null(), &
            logl_r_pgstar_decorator => null(), &
            logl_teff_pgstar_decorator => null(), &
            logl_v_pgstar_decorator => null(), &
            l_r_pgstar_decorator => null(), &
            l_teff_pgstar_decorator => null(), &
            l_v_pgstar_decorator => null(), &
            mixing_pgstar_decorator => null(), &
            mode_prop_pgstar_decorator => null(), &
            network_pgstar_decorator => null(), &
            power_pgstar_decorator => null(), &
            production_pgstar_decorator => null(), &
            Profile_Panels1_pgstar_decorator => null(), &
            Profile_Panels2_pgstar_decorator => null(), &
            Profile_Panels3_pgstar_decorator => null(), &
            Profile_Panels4_pgstar_decorator => null(), &
            Profile_Panels5_pgstar_decorator => null(), &
            Profile_Panels6_pgstar_decorator => null(), &
            Profile_Panels7_pgstar_decorator => null(), &
            Profile_Panels8_pgstar_decorator => null(), &
            Profile_Panels9_pgstar_decorator => null(), &
            R_L_pgstar_decorator => null(), &
            R_Teff_pgstar_decorator => null(), &
            rti_pgstar_decorator => null(), &
            summary_burn_pgstar_decorator => null(), &
            summary_profile_pgstar_decorator => null(), &
            summary_history_pgstar_decorator => null(), &
            trho_pgstar_decorator => null(), &
            trho_profile_pgstar_decorator => null()
