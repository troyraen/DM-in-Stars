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

      module const_def
      implicit none


      ! real number precision options: single, double, quad
      integer, parameter :: sp = selected_real_kind(p=5)
      integer, parameter :: dp = selected_real_kind(p=15)
      integer, parameter :: qp = selected_real_kind(p=30)

      ! integer precision options
      integer, parameter :: i4 = selected_int_kind(9)
      integer, parameter :: i8 = selected_int_kind(14)


      integer, parameter :: strlen = 256 ! for character (len=strlen)


!
! mathematical and physical constants (in cgs)
!
! the 2006 CODATA recommended values of the physical constants
! by cohen & taylor



! math constants
      real(dp), parameter :: pi = 3.1415926535897932384626433832795028841971693993751d0
      real(dp), parameter :: pi4 = 4*pi
      real(dp), parameter :: eulercon = 0.577215664901532861d0
      real(dp), parameter :: ln2 = 6.9314718055994529D-01 ! = log_cr(2d0)
      real(dp), parameter :: ln3 = 1.0986122886681096D+00 ! = log_cr(3d0)
      real(dp), parameter :: ln10 = 2.3025850929940455D+00 ! = log_cr(10d0)
      real(dp), parameter :: a2rad = pi/180.0d0 ! angle to radians
      real(dp), parameter :: rad2a = 180.0d0/pi ! radians to angle
      real(dp), parameter :: one_third = 1d0/3d0
      real(dp), parameter :: two_thirds = 2d0/3d0
      real(dp), parameter :: ln4pi3 = 1.4324119583011810d0 ! = log_cr(4*pi/3)
      real(dp), parameter :: two_13 = 1.2599210498948730d0 ! = pow_cr(2d0,1d0/3d0)
      real(dp), parameter :: four_13 = 1.5874010519681994d0 ! = pow_cr(4d0,1d0/3d0)


! physical constants
      real(dp) :: standard_cgrav ! = 6.67428d-8
         ! gravitational constant (g^-1 cm^3 s^-2)
      real(dp) :: planck_h ! = 6.62606896D-27
         ! Planck's constant (erg s)
      real(dp) :: hbar ! = planck_h / (2*pi)
      real(dp) :: qe ! = 4.80320440D-10
         ! electron charge (esu == (g cm^3 s^-2)^(1/2))
      real(dp) :: avo ! = 6.02214179d23
         ! Avogadro's constant (mole^-1)
      real(dp) :: clight ! = 2.99792458d10
         ! speed of light in vacuum (cm s^1)
      real(dp) :: kerg ! = 1.3806504D-16
         ! Boltzmann's constant (erg K^-1)
      real(dp) :: boltzm ! = kerg
      real(dp) :: cgas ! = boltzm*avo ! R_gas; ideal gas constant; erg/K/mole
      real(dp) :: kev ! = 8.617385d-5
         ! converts temp (kelvin) to ev (ev K^-1)
      real(dp) :: amu ! = 1.660538782d-24
         ! atomic mass unit (g)

      real(dp) :: mn ! = 1.6749286d-24 ! neutron mass (g)
      real(dp) :: mp ! = 1.6726231d-24 ! proton mass (g)
      real(dp) :: me ! = 9.1093826D-28 ! (was 9.1093897d-28) electron mass (g)
      real(dp) :: rbohr ! = hbar*hbar/(me * qe * qe) ! Bohr radius (cm)
      real(dp) :: fine ! = qe*qe/(hbar*clight) ! fine structure constant
      real(dp) :: hion ! = 13.605698140d0 ! hydrogen ionization energy (eV)
      real(dp) :: ev2erg ! = 1.602176487d-12 ! electron volt (erg)
      real(dp) :: mev_to_ergs ! = 1d6*ev2erg
      real(dp) :: mev_amu ! = mev_to_ergs/amu
      real(dp) :: Qconv ! = ev2erg*1.0d6*avo; convert Q rates to erg/gm/sec

      real(dp) :: boltz_sigma ! = 5.670400D-5
         ! boltzmann's sigma = crad*clight/4 (erg cm^-2 K^-4 s^-1)
      real(dp) :: crad ! = boltz_sigma*4/clight, 7.5657673816464059d-15
         ! radiation density constant, a (erg cm^-3 K^-4); Prad ! = crad * T^4 / 3

      real(dp) :: ssol ! = boltz_sigma
      real(dp) :: asol ! = crad
      real(dp) :: weinlam ! = planck_h*clight/(kerg * 4.965114232d0)
      real(dp) :: weinfre ! = 2.821439372d0*kerg/planck_h
      real(dp) :: rhonuc ! = 2.342d14 ! density of nucleus (g cm^3)


! astronomical constants
      ! solar age, L, and R values from Bahcall et al, ApJ 618 (2005) 1049-1056.
      real(dp) :: msol ! = 1.9892d33  ! solar mass (g)
      real(dp) :: rsol ! = 6.9598d10 ! solar radius (cm)
      real(dp) :: lsol ! = 3.8418d33  ! solar luminosity (erg s^-1)
      real(dp) :: agesol ! = 4.57d9  ! solar age (years)
      real(dp) :: Msun ! = msol
      real(dp) :: Rsun ! = rsol
      real(dp) :: Lsun ! = lsol
      real(dp) :: Msun33 ! = msol*1d-33
      real(dp) :: Rsun11 ! = rsol*1d-11
      real(dp) :: Lsun33 ! = lsol*1d-33
      real(dp) :: Teffsol ! = 5777d0 ! temperature (k)
      real(dp) :: loggsol ! = 4.4378893534131256d0 ! log surface gravity ! log(g/(cm s^-2))
      real(dp) :: teffsun ! = teffsol
      real(dp) :: loggsun ! = loggsol
      real(dp) :: mbolsun ! = 4.74 ! Bolometric magnitude of the Sun
      real(dp) :: mbolsol ! = mbolsun

      real(dp) :: ly ! = 9.460528d17 ! light year (cm)
      real(dp) :: pc ! = 3.261633d0 * ly ! parsec (cm)
      real(dp) :: secyer ! 3.1558149984d7 ! seconds per year
      real(dp) :: dayyer ! 365.25 ! days per year

      real(dp) :: m_earth ! = 5.9764d27 ! earth mass (g)
      real(dp) :: r_earth ! = 6.37d8 ! earth radius (cm)
      real(dp) :: au ! = 1.495978921d13 ! astronomical unit (cm)

      real(dp) :: m_jupiter ! = 1.8986d30 ! jupiter mass (g)
      real(dp) :: r_jupiter ! = 6.9911d9 ! jupiter mean radius (cm)
      real(dp) :: semimajor_axis_jupiter ! = 7.7857d13 ! jupiter semimajor axis (cm)


      ! many routines allow either a value, a log value, or both as args
      ! omitted args are indicated by passing 'arg_not_provided'

      real(dp), parameter :: arg_not_provided = -9d99
      real(dp), parameter :: missing_value = arg_not_provided

      character (len=strlen) :: mesa_dir
      character (len=strlen) :: mesa_data_dir ! = trim(mesa_dir) // '/data'
      character (len=strlen) :: mesa_caches_dir
      character (len=strlen) :: mesa_temp_caches_dir !Temp storage must be local to run

      logical :: use_mesa_temp_cache ! If false then dont use mesa_temp_caches_dir

      ! mixing types
      ! NOTE: some packages may depend on the order
      integer, parameter :: no_mixing = 0
      integer, parameter :: convective_mixing = 1
      integer, parameter :: softened_convective_mixing = 2
         ! for modified D_mix near convective boundary
      integer, parameter :: overshoot_mixing = 3
      integer, parameter :: semiconvective_mixing = 4
      integer, parameter :: thermohaline_mixing = 5
      integer, parameter :: rotation_mixing = 6
      integer, parameter :: rayleigh_taylor_mixing = 7
      integer, parameter :: minimum_mixing = 8
      integer, parameter :: anonymous_mixing = 9


      contains


      subroutine initialize_const_vals

         standard_cgrav = 6.67428d-8
            ! gravitational constant (g^-1 cm^3 s^-2)
         planck_h = 6.62606896D-27
            ! Planck's constant (erg s)
         hbar = planck_h / (2*pi)
         qe = 4.80320440D-10
            ! electron charge (esu == (g cm^3 s^-2)^(1/2))
         avo = 6.02214179d23
            ! Avogadro's constant (mole^-1)
         clight = 2.99792458d10
            ! speed of light in vacuum (cm s^-1)
         kerg = 1.3806504D-16
            ! Boltzmann's constant (erg K^-1)
         boltzm = kerg
         cgas = boltzm*avo !ideal gas constant; erg/K
         kev = 8.617385d-5
            ! converts temp to ev (ev K^-1)
         amu = 1d0/avo
            ! atomic mass unit (g)

         mn = 1.6749286d-24 ! neutron mass (g)
         mp = 1.6726231d-24 ! proton mass (g)
         me = 9.1093826D-28 ! (was 9.1093897d-28) electron mass (g)
         rbohr = hbar*hbar/(me * qe * qe) ! Bohr radius (cm)
         fine = qe*qe/(hbar*clight) ! fine structure constant
         hion = 13.605698140d0 ! hydrogen ionization energy (eV)
         ev2erg = 1.602176487d-12 ! electron volt (erg)
         mev_to_ergs = 1d6*ev2erg
         mev_amu = mev_to_ergs/amu
         Qconv = mev_to_ergs*avo

         boltz_sigma = 5.670400D-5
            ! boltzmann's sigma = a*c/4 (erg cm^-2 K^-4 s^-1)
         crad = boltz_sigma*4/clight
            ! = radiation density constant, a (erg cm^-3 K^-4); Prad = crad * T^4 / 3
            ! approx = 7.5657e-15

         ssol = boltz_sigma
         asol = crad
         weinlam = planck_h*clight/(kerg * 4.965114232d0)
         weinfre = 2.821439372d0*kerg/planck_h
         rhonuc = 2.342d14 ! density of nucleus (g cm^3)

         ! solar age, L, and R values from Bahcall et al, ApJ 618 (2005) 1049-1056.
         msol = 1.9892d33  ! solar mass (g)  <<< gravitational mass, not baryonic
         rsol = 6.9598d10 ! solar radius (cm)
         lsol = 3.8418d33  ! solar luminosity (erg s^-1)
         agesol = 4.57d9  ! solar age (years)
         Msun = msol
         Rsun = rsol
         Lsun = lsol
         Msun33 = msol*1d-33
         Rsun11 = rsol*1d-11
         Lsun33 = lsol*1d-33
         ly = 9.460528d17 ! light year (cm)
         pc = 3.261633d0 * ly ! parsec (cm)
         secyer = 3.1558149984d7 ! seconds per year
         dayyer = 365.25d0 ! days per year

         Teffsol=5777.0d0
         loggsol=4.4378893534131256d0 ! With mesa's default msol, rsol and standard_cgrav
         Teffsun=teffsol
         loggsun=loggsol

         mbolsun = 4.74d0  ! Bolometric magnitude of the Sun IAU Resolution B1 2015
         mbolsol = mbolsun

         m_earth = 5.9764d27 ! earth mass (g)
            ! = 3.004424e-6 Msun
         r_earth = 6.37d8 ! earth radius (cm)
         au = 1.495978921d13 ! astronomical unit (cm)

         m_jupiter = 1.8986d30 ! jupiter mass (g)
            ! = 0.954454d-3 Msun
         r_jupiter = 6.9911d9 ! jupiter mean radius (cm)
         semimajor_axis_jupiter = 7.7857d13 ! jupiter semimajor axis (cm)

      end subroutine initialize_const_vals


      subroutine do_const_init(mesa_dir_init, ierr)
         character (len=*), intent(in) :: mesa_dir_init
         integer, intent(out) :: ierr

         integer :: i, iounit, version_number
         character (len=strlen) :: filename, temp_caches_disable

         ierr = 0

         call initialize_const_vals

         call get_environment_variable("MESA_CACHES_DIR", mesa_caches_dir)
         !write(*,*) 'MESA_CACHES_DIR "' // trim(mesa_caches_dir) // '"'

         mesa_dir = mesa_dir_init
         if (len_trim(mesa_dir) == 0) then
            call get_environment_variable("MESA_DIR", mesa_dir)
         end if

         if (len_trim(mesa_dir) > 0) then
            mesa_data_dir = trim(mesa_dir) // '/data'
         else
            write(*,*) 'ERROR: you must provide the path to your mesa directory,'
            write(*,*) 'either in your inlist or by setting the MESA_DIR environment variable.'
            ierr = -1
            return
         end if

         !write(*,*) 'mesa_data_dir ' // trim(mesa_data_dir)

         call get_environment_variable("MESA_TEMP_CACHES_DIR", mesa_temp_caches_dir)

         if (len_trim(mesa_temp_caches_dir) == 0) then
            mesa_temp_caches_dir = './.mesa_temp_cache'
         end if

         !Opt out for temp caches
         use_mesa_temp_cache=.true.
         call get_environment_variable("MESA_TEMP_CACHES_DISABLE", temp_caches_disable)

         if (len_trim(temp_caches_disable) > 0) then
            use_mesa_temp_cache = .false.
         end if

         iounit = 33
         filename = trim(mesa_data_dir) // '/version_number'
         open(unit=iounit, file=trim(filename), status='old', action='read', iostat=ierr)
         if (ierr /= 0) then
            write(*,*) 'open failed for ' // trim(filename)
            write(*,*) 'please check that mesa_dir is set correctly'
            return
         end if
         read(iounit,*,iostat=ierr) version_number
         close(iounit)
         if (ierr /= 0) then
            write(*, '(a)') 'failed to read version number in ' // trim(filename)
            write(*,*) 'please check that mesa_dir is set correctly'
            return
         end if

         if (version_number <= 0 .or. version_number > 100000) then
            write(*, '(a,i6)') 'invalid version number in ' // trim(filename), version_number
            write(*,*) 'please check that mesa_dir is set correctly'
            ierr = -1
            return
         end if

         !write(*,*) 'version_number', version_number

      end subroutine do_const_init


      end module const_def
