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
!
! ***********************************************************************

      module atm_def
      
      use const_def, only: dp
      
      implicit none
      
#ifdef offload
      !dir$ options /offload_attribute_target=mic
#endif      

      integer, parameter :: atm_simple_photosphere = 1
      integer, parameter :: atm_Eddington_grey = 2 ! Eddington T-tau integration
      integer, parameter :: atm_Krishna_Swamy = 3 ! Krishna Swamy T-tau integration
      integer, parameter :: atm_solar_Hopf_grey = 4
         ! solar calibrated Hopf-function T-tau integration
         ! T^4 = 3/4 Teff^4 (tau + q(tau))
         ! q(tau) = q1 + q2 exp(-q3 tau) + q4 exp(-q5 tau)
         ! solar calibrated q's (from Jorgen Christensen-Dalsgaard) are
         !     q1 = 1.0361
         !     q2 = -0.3134 
         !     q3 = 2.44799995
         !     q4 = -0.29589999
         !     q5 = 30.0
         ! tau_photoshere is tau s.t. tau + q(tau) = 4/3 => tau_photosphere = 0.4116433502
         
      integer, parameter :: atm_tau_100_tables = 5
         ! use model atmosphere tables for Pgas and T at tau=100; solar Z only.
      integer, parameter :: atm_tau_10_tables = 6
         ! use model atmosphere tables for Pgas and T at tau=10; solar Z only.
      integer, parameter :: atm_tau_1_tables = 7 
         ! use model atmosphere tables for Pgas and T at tau=1; solar Z only.
      integer, parameter :: atm_tau_1m1_tables = 8
         ! use model atmosphere tables for Pgas and T at tau=1e-1; solar Z only.
                  
      integer, parameter :: atm_photosphere_tables = 9 
         ! use model atmosphere tables for photosphere Pgas; [Z/Z_SOLAR] from -4.0 to +0.5
      integer, parameter :: atm_grey_and_kap = 10 ! find consistent P, T, and kap at surface
      integer, parameter :: atm_grey_irradiated = 11  
         ! based on Guillot, T, and Havel, M., A&A 527, A20 (2011). see equation 6.
      integer, parameter :: atm_Paczynski_grey = 12
         ! integrate an atmosphere for given base conditions.
         ! inspired by B. Paczynski, 1969, Acta Astr., vol. 19, 1.
         ! takes into account dilution when tau < 2/3,
         ! and calls mlt to get gradT allowing for convection.
         ! note: only available from mesa/star since requires star lib information
      integer, parameter :: atm_WD_tau_25_tables = 13
         ! hydrogen atmosphere tables for cool white dwarfs
         ! giving Pgas and T at log10(tau) = 1.4 (tau = 25.11886)
         ! Teff goes from 40,000 K down to 2,000K with step of 100 K
         ! Log10(g) goes from 9.5 down to 5.5 with step of 0.1 
         ! reference
            ! R.D. Rohrmann, L.G. Althaus, and S.O. Kepler,
            ! Lyman α wing absorption in cool white dwarf stars,
            ! Mon. Not. R. Astron. Soc. 411, 781–791 (2011)
      integer, parameter :: atm_fixed_Teff = 14
         ! set Tsurf from Eddington T-tau relation for given Teff and tau
         ! set Psurf = Radiation_Pressure(Tsurf)
      integer, parameter :: atm_fixed_Tsurf = 15
         ! set Teff from Eddington T-tau relation for given Tsurf and tau=2/3
         ! set Psurf = Radiation_Pressure(Tsurf)
      integer, parameter :: atm_fixed_Psurf = 16
         ! get value of Psurf from control parameter use_this_fixed_Psurf.
         ! set Tsurf from L and R using L = 4*pi*R^2*boltz_sigma*T^4.
         ! set Teff using Eddington T-tau relation for tau=2/3 and T=Tsurf.

      integer, parameter :: min_atm_option = 1 
      integer, parameter :: max_atm_option = 16
      
      
      ! info about structure of atmosphere
      integer, parameter :: atm_xm = 1 ! mass of atm exterior to this point (g)
      integer, parameter :: atm_delta_r = atm_xm+1 ! radial distance above base of envelope (cm)
      integer, parameter :: atm_lnP = atm_delta_r+1
      integer, parameter :: atm_lnd = atm_lnP+1
      integer, parameter :: atm_lnT = atm_lnd+1
      integer, parameter :: atm_gradT = atm_lnT+1
      integer, parameter :: atm_kap = atm_gradT+1
      integer, parameter :: atm_gamma1 = atm_kap+1
      integer, parameter :: atm_grada = atm_gamma1+1
      integer, parameter :: atm_chiT = atm_grada+1
      integer, parameter :: atm_chiRho = atm_chiT+1
      integer, parameter :: atm_cv = atm_chiRho+1
      integer, parameter :: atm_cp = atm_cv+1
      integer, parameter :: atm_lnfree_e = atm_cp+1
      integer, parameter :: atm_dlnkap_dlnT = atm_lnfree_e+1
      integer, parameter :: atm_dlnkap_dlnd = atm_dlnkap_dlnT+1
      integer, parameter :: atm_lnPgas = atm_dlnkap_dlnd+1
      integer, parameter :: atm_tau = atm_lnPgas+1
      integer, parameter :: atm_gradr = atm_tau+1

      integer, parameter :: num_results_for_create_atm = atm_gradr 
      
      
      ! tables
      integer, parameter :: table_atm_version = 6
      
      type Atm_Info
         integer :: which_atm_option, nZ, ng, nT, ilinT, iling
         real(dp), pointer :: Teff_array(:), logg_array(:), Teff_bound(:)
         real(dp), pointer :: logZ(:), alphaFe(:)
         real(dp), pointer :: Pgas_interp1(:), T_interp1(:)
         real(dp), pointer :: Pgas_interp(:,:,:,:), T_interp(:,:,:,:)
         character(len=8), pointer :: atm_mix(:)
         character(len=40), pointer :: table_atm_files(:)
         logical, pointer :: have_atm_table(:)
      end type Atm_Info
      
      type (Atm_Info), target :: &
         ai_two_thirds_info, ai_100_info, ai_10_info, ai_1_info, &
         ai_1m1_info, ai_wd_25_info
      type (Atm_Info), pointer :: &
         ai_two_thirds, ai_100, ai_10, ai_1, ai_1m1, ai_wd_25


      ! pairs of x and log10[ExpIntegralE[2,x]] from Mathematica
      integer, parameter :: npairs = 571
      real(dp), target :: E2_x(npairs)
      real(dp) :: E2_pairs(2*npairs)
      real(dp), target :: E2_f_ary(4*npairs)
      real(dp), pointer :: E2_f1(:), E2_f(:,:)
      logical :: have_E2_interpolant = .false.

      logical :: table_atm_is_initialized = .false.
      

      type Int_Atm_Info
         logical :: save_atm_structure_info
         integer :: atm_structure_num_pts
         real(dp), pointer :: atm_structure(:,:) ! will be allocated if necessary
            ! (num_results_for_create_atm, num_atm_points)
         ! bookkeeping
         integer :: handle
         logical :: in_use
      end type Int_Atm_Info
      
      logical :: int_atm_is_initialized = .false.
      integer, parameter :: max_atm_handles = 10
      type (Int_Atm_Info), target :: atm_handles(max_atm_handles)


      contains
      
      
      subroutine copy_atm_info_to_coprocessor(ierr) ! runs on host
         integer, intent(out) :: ierr
         real(dp), pointer :: E2_x_in(:)
         real(dp), pointer :: E2_f1_in(:)
         ierr = 0
#ifdef offload
         if (.not. have_E2_interpolant) then
            write(*,*) 'copy_atm_info_to_coprocessor: no E_interpolant'
            stop 1
         end if
         E2_x_in => E2_x
         E2_f1_in => E2_f1
         !dir$ offload target(mic) out(ierr) in(E2_x_in, E2_f1_in)
         call do_copy_atm_info_to_coprocessor( &
            E2_x_in, E2_f1_in, ierr)
#endif
      end subroutine copy_atm_info_to_coprocessor
      
#ifdef offload
      subroutine do_copy_atm_info_to_coprocessor( & ! runs on mic
            E2_x_in, E2_f1_in, ierr)
         real(dp) :: E2_x_in(npairs)
         real(dp), pointer :: E2_f1_in(:)
         integer, intent(out) :: ierr
         
         integer :: i
         
         ierr = 0

         call set_E2_pairs

         E2_f1 => E2_f_ary
         E2_f(1:4,1:npairs) => E2_f1(1:4*npairs)
         
         do i=1,4*npairs
            E2_f1(i) = E2_f1_in(i)
         end do
         
         do i=1,npairs
            E2_x(i) = E2_x_in(i)
         end do
      
         ai_two_thirds => ai_two_thirds_info
         ai_100 => ai_100_info
         ai_10 => ai_10_info
         ai_1 => ai_1_info
         ai_1m1 => ai_1m1_info
         ai_wd_25 => ai_wd_25_info
         
         int_atm_is_initialized = .true.
         table_atm_is_initialized = .true.
         
      end subroutine do_copy_atm_info_to_coprocessor
#endif
      
      subroutine copy_atm_table_to_coprocessor(ai,iZ,ierr) ! runs on host
         type (Atm_Info), pointer :: ai
         integer, intent(in) :: iZ
         integer, intent(out) :: ierr
         
         integer :: which_atm_option, nZ, ng, nT, ilinT, iling
         real(dp), pointer, dimension(:) :: &
            Teff_array, logg_array, Teff_bound, logZ, alphaFe, Pgas_f1, T_f1
         character(len=8), pointer :: atm_mix(:)
         character(len=40), pointer :: table_atm_files(:)
         logical, pointer :: have_atm_table(:)
         
         ierr = 0
#ifdef offload
         nZ = ai% nZ
         nT = ai% nT
         ng = ai% ng
         which_atm_option = ai% which_atm_option
         ilinT = ai% ilinT
         iling = ai% iling
         Teff_array => ai% Teff_array
         logg_array => ai% logg_array
         Teff_bound => ai% Teff_bound
         logZ => ai% logZ
         alphaFe => ai% alphaFe
         atm_mix => ai% atm_mix
         table_atm_files => ai% table_atm_files
         have_atm_table => ai% have_atm_table
         Pgas_f1(1:4*ng*nT) => ai% Pgas_interp1(1+4*ng*nT*(iZ-1):4*ng*nT*iZ)
         T_f1(1:4*ng*nT) => ai% T_interp1(1+4*ng*nT*(iZ-1):4*ng*nT*iZ)
      
         !dir$ offload target(mic) out(ierr) in( &
            which_atm_option, iZ, nZ, ng, nT, ilinT, iling, &
            Teff_array, logg_array, Teff_bound, logZ, alphaFe, Pgas_f1, T_f1, &
            atm_mix, table_atm_files)
         call do_copy_atm_table_to_coprocessor( & ! runs on mic
            which_atm_option, iZ, nZ, ng, nT, ilinT, iling, &
            Teff_array, logg_array, Teff_bound, logZ, alphaFe, Pgas_f1, T_f1, &
            atm_mix, table_atm_files, have_atm_table, ierr)
             
#endif
      end subroutine copy_atm_table_to_coprocessor

#ifdef offload
      subroutine do_copy_atm_table_to_coprocessor( & ! runs on mic
            which_atm_option, iZ, nZ, ng, nT, ilinT, iling, &
            Teff_array, logg_array, Teff_bound, logZ, alphaFe, Pgas_f1, T_f1, &
            atm_mix, table_atm_files, have_atm_table, ierr)
         integer, intent(in) :: which_atm_option, iZ, nZ, ng, nT, ilinT, iling
         real(dp), pointer, dimension(:), intent(in) :: &
            Teff_array, logg_array, Teff_bound, logZ, alphaFe, Pgas_f1, T_f1
         character(len=8), pointer, intent(in) :: atm_mix(:)
         character(len=40), pointer, intent(in) :: table_atm_files(:)
         logical, pointer, intent(in) :: have_atm_table(:)
         integer, intent(out) :: ierr

         type (Atm_Info), pointer :: ai
         real(dp), pointer :: f1(:)
         integer :: i, sz
         
         include 'formats'

         ierr = 0
         
         if (which_atm_option == atm_photosphere_tables) then
            ai => ai_two_thirds
         else if (which_atm_option == atm_tau_100_tables) then
            ai => ai_100
         else if (which_atm_option == atm_tau_10_tables) then
            ai => ai_10
         else if (which_atm_option == atm_tau_1_tables) then
            ai => ai_1
         else if (which_atm_option == atm_tau_1m1_tables) then
            ai => ai_1m1
         else if (which_atm_option == atm_wd_tau_25_tables) then
            ai => ai_wd_25
         else
            write(*,*) 'bad value for which_atm_option', which_atm_option
            ierr = -1
            return 
         end if
         
         ai% nZ = nZ
         ai% nT = nT
         ai% ng = ng
         ai% which_atm_option = which_atm_option
         ai% ilinT = ilinT
         ai% iling = iling
         
         if (.not. associated(ai% have_atm_table)) then
            sz = size(have_atm_table,dim=1)
            allocate(ai% have_atm_table(sz))
            do i=1,sz
               ai% have_atm_table(i) = have_atm_table(i)
            end do
         end if
         ai% have_atm_table(iZ) = .true.
         
         if (.not. associated(ai% Teff_array)) then
            allocate(ai% Teff_array(nT))
            do i=1,nT
               ai% Teff_array(i) = Teff_array(i)
            end do
         end if
         
         if (.not. associated(ai% logg_array)) then
            allocate(ai% logg_array(ng))
            do i=1,ng
               ai% logg_array(i) = logg_array(i)
            end do
         end if
         
         if (.not. associated(ai% Teff_bound)) then
            allocate(ai% Teff_bound(ng))
            do i=1,ng
               ai% Teff_bound(i) = Teff_bound(i)
            end do
         end if
         
         if (.not. associated(ai% logZ)) then
            allocate(ai% logZ(nZ))
            do i=1,nZ
               ai% logZ(i) = logZ(i)
            end do
         end if
         
         if (.not. associated(ai% alphaFe)) then
            allocate(ai% alphaFe(nZ))
            do i=1,nZ
               ai% alphaFe(i) = alphaFe(i)
            end do
         end if

         if (.not. associated(ai% atm_mix)) then
            allocate(ai% atm_mix(nZ))
            do i=1,nZ
               ai% atm_mix(i) = atm_mix(i)
            end do
         end if
         
         if (.not. associated(ai% table_atm_files)) then
            allocate(ai% table_atm_files(nZ))
            do i=1,nZ
               ai% table_atm_files(i) = table_atm_files(i)
            end do
         end if

         if (.not. associated(ai% Pgas_interp1)) then
            allocate(ai% Pgas_interp1(4*ng*nT*nZ))
            ai% Pgas_interp(1:4,1:ng,1:nT,1:nZ) => ai% Pgas_interp1(1:4*ng*nT*nZ)
         end if
         
         f1(1:4*ng*nT) => ai% Pgas_interp1(1+4*ng*nT*(iZ-1):4*ng*nT*iZ)
         do i=1,4*ng*nT
            f1(i) = Pgas_f1(i)
         end do
         
         if (.not. associated(ai% T_interp1)) then
            allocate(ai% T_interp1(4*ng*nT*nZ))
            ai% T_interp(1:4,1:ng,1:nT,1:nZ) => ai% T_interp1(1:4*ng*nT*nZ)
         end if
         
         f1(1:4*ng*nT) => ai% T_interp1(1+4*ng*nT*(iZ-1):4*ng*nT*iZ)
         do i=1,4*ng*nT
            f1(i) = T_f1(i)
         end do

      end subroutine do_copy_atm_table_to_coprocessor
#endif
      
      subroutine set_E2_pairs 
         include 'e2_pairs.dek'
      end subroutine set_E2_pairs

#ifdef offload
      !dir$ end options
#endif

      end module atm_def

