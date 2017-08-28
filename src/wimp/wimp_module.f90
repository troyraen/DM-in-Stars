!!!-------------------------------------------!!!
!!!	Troy Joseph Raen
!!!	June 2017
!!!
!!!	add-on module for MESA
!!!	calculates extra heat transported by WIMPS
!!!-------------------------------------------!!!
!!! controls:
!!! Nx = s% xtra1
!!!	cboost = s% x_ctrl(1)
!!!	spindep = s% x_logical_ctrl(1)  ! .true. = spin dependent; .false. = spin independent


	MODULE wimp_module

	use star_def
	use const_def
	use chem_def
	use wimp_num
	IMPLICIT NONE
	INCLUDE 'wimp_vars.h'

	CONTAINS


!!----------------------------
!!	main routine, called by MESA run_star_extras.f
!!----------------------------
	SUBROUTINE wimp_energy_transport(id,ierr)
	IMPLICIT NONE
	!INCLUDE 'wimp_vars.h'

	INTEGER, INTENT(IN) :: id
	INTEGER, INTENT(OUT) :: ierr
	INTEGER :: itr

	TYPE (star_info), pointer :: s ! pointer to star type
	ierr=0
	CALL GET_STAR_PTR(id, s, ierr)
	IF ( ierr /= 0 ) RETURN

	spindep = s% x_logical_ctrl(1)
	CALL get_star_variables(id,ierr)
	CALL set_wimp_variables(id,ierr)
	CALL calc_xheat()


	!! start after 10,000 years
	IF ( Age_star .LT. 1.D4) THEN
		DO itr = 1,kmax
			s% extra_heat(itr) = 0.D0
		ENDDO
	ELSE
		DO itr = 1,kmax
			s% extra_heat(itr) = xheat(itr)
		ENDDO
	ENDIF

	END SUBROUTINE wimp_energy_transport


!!----------------------------
!!	grab star variables from MESA
!!	calculate vesc and V(k)
!!----------------------------
	SUBROUTINE get_star_variables(id,ierr)
	use const_def, only : mp, Rsun, standard_cgrav, amu
	! proton mass (g), solar radius (cm), Grav const (g^-1 cm^3 s^-2), amu (g)
	IMPLICIT NONE
	!INCLUDE 'wimp_vars.h'

	INTEGER, INTENT(IN) :: id
	INTEGER, INTENT(OUT) :: ierr
	INTEGER :: itr, j
	TYPE (star_info), pointer :: s ! pointer to star type
	ierr=0
	CALL GET_STAR_PTR(id, s, ierr)
	IF ( ierr /= 0 ) RETURN

	! star variables
	Age_star = s% star_age !! in years
	dttmp = s% dt !! in seconds
	maxT = 10.D0**(s% log_max_temperature) !! max temp in K
	M_star = s% mstar !! in grams
	R_star = (s% photosphere_r)* Rsun !! convert to cm
	vesc = SQRT(2.D0* standard_cgrav* M_star/ R_star)

	numspecies = s% species
	WRITE(*,*) 'numspecies=',numspecies, 'maxspecies=',maxspecies
	IF (numspecies .GT. maxspecies) THEN
		WRITE(*,*) '*** numspecies > maxspecies = ',maxspecies
		WRITE(*,*) '**** STOPPING RUN AT star_age = ',Age_star,' years'
		STOP
	ENDIF

	! copy cell variables
	kmax = s% nz
	IF ((kmax+1) .GT. maxcells) THEN
		WRITE(*,*) '**** kmax+1 > maxcells = ',maxcells
		WRITE(*,*) '**** STOPPING RUN AT star_age = ',Age_star,' years'
		STOP
	ENDIF

	DO itr = 1,kmax
		Xk(itr) = s% X(itr) !! mass fraction hydrogen
		Tk(itr) = s% T(itr) !! in K
		rhok(itr) = s% rho(itr) !! in g/cm^3
		npk(itr) = Xk(itr)*rhok(itr)/mp !! cm^-3
		rk(itr) = s% r(itr) !! in cm
		gravk(itr) = s% grav(itr) !! in cm/s^2
		! info on all elements in the net:
		IF (.NOT. spindep) THEN
			DO j = 1,numspecies
				chemj = s% chem_id(j) !! gives index of element j in chem_isos
				IF (itr .EQ. 1) THEN
					mj(j) = chem_isos% W(chemj) * amu ! mass of element j (g)
					mGeVj(j) = mj(j)/gperGeV ! mass in GeV
					Aj(j) = chem_isos% Z_plus_N(chemj) ! mass number of element j
					WRITE(*,*) chem_isos% name(chemj), mGeVj(j), Aj(j)
				ENDIF
				xajk(j,itr) = s% xa(j,itr) ! mass fraction of element j in cell k
				njk(j,itr) = xajk(j,itr)*rhok(itr)/mj(j) ! number fraction of element j in cell k
			ENDDO
		ENDIF
	ENDDO

	! set central values (none set for elements beyond H1)
	Xk(kmax+1) = s% center_h1
	Tk(kmax+1) = 10.D0**(s% log_center_temperature)
	rhok(kmax+1) = 10.D0**(s% log_center_density)
	npk(kmax+1) = Xk(kmax+1)*rhok(kmax+1)/mp
	rk(kmax+1) = 0.D0
	gravk(kmax+1) = 0.D0
	Vk(kmax+1) = 0.D0

	! calculate V(k) from center to outer edge
	DO itr = kmax, 1, -1
		Vk(itr) = Vk(itr+1)+ 0.5D0*(gravk(itr)+ gravk(itr+1))* (rk(itr)- rk(itr+1))
	ENDDO

	END SUBROUTINE get_star_variables




!!----------------------------
!!	set all WIMP properties
!!----------------------------
	SUBROUTINE set_wimp_variables(id,ierr)
	IMPLICIT NONE
	!INCLUDE 'wimp_vars.h'
	DOUBLE PRECISION :: dNx

	INTEGER, INTENT(IN) :: id
	INTEGER, INTENT(OUT) :: ierr
	INTEGER :: itr, j
	TYPE (star_info), pointer :: s ! pointer to star type
	ierr=0
	CALL GET_STAR_PTR(id, s, ierr)
	IF ( ierr /= 0 ) RETURN

	mxGeV = 5.D0	! 5 GeV WIMP
	mx = mxGeV* gperGeV	! WIMP mass in grams
	cboost = s% x_ctrl(1)  ! boost in capture rate of WIMPs compared to the local capture rate near the Sun, \propto density/sigma_v
	IF (spindep) THEN
		sigmaxp = 1.D-37	! wimp-proton cross section, cm^2
	ELSE
		sigmaxp = 1.D-40
		sigmaX = 0.D0
		DO j = 1,numspecies
			sigmaxj(j) = sigmaxp* (Aj(j)*mj(j)/mp)**2 * ((mx+mp)/(mx+mj(j)))**2
			sigmaX = sigmaX+ sigmaxj(j)
		ENDDO
	ENDIF

	Tx = calc_Tx()
	dNx = calc_dNx()
	s% xtra1 = (s% xtra1) + dNx
	Nx = s% xtra1
	CALL calc_nxk()

	END SUBROUTINE set_wimp_variables



!!----------------------------
!!	xheat is luminosity per unit mass (in cgs units) that wimps give up to baryons
!!	This is what needs to be given to the star program
!!	This is SP85 equ 4.9 divided by rho(k)
!!----------------------------
	SUBROUTINE calc_xheat()
	USE const_def, only : pi, mp, kerg ! Boltzmann's constant (erg K^-1)
	IMPLICIT NONE
	INTEGER :: itr, j
	DOUBLE PRECISION :: mfact, dfact, Tfact

!	LOGICAL :: ISOPEN

!	INQUIRE(FILE='xheat.txt', OPENED=ISOPEN)
!	IF (.NOT. ISOPEN) THEN
!		OPEN(FILE='xheat.txt', UNIT=10)
!	ENDIF

	IF (spindep) THEN
		mfact = 8.D0*SQRT(2.D0/pi)* sigmaxp* mx*mp/((mx+mp)**2) ! this is common to all cells

		DO itr = 1,kmax
			dfact = nxk(itr)*npk(itr)/rhok(itr)
			Tfact = SQRT((mp*kerg*Tx+ mx*kerg*Tk(itr))/(mx*mp)) * kerg*(Tx- Tk(itr))	! Tx-Tk gives correct sign
			xheat(itr) = mfact* dfact* Tfact
	!		WRITE(10,*) itr, xheat(itr)
		ENDDO

	!	DO itr = 1,kmax
	!		WRITE(10,"(F10.5)",advance="no") xheat(itr)
	!	ENDDO

	ELSE
		DO itr = 1,kmax
			xheat(itr) = 0.D0
			DO j = 1,numspecies
				mfact = 8.D0*SQRT(2.D0/pi)* sigmaxj(j)* mx*mj(j)/((mx+mj(j))**2)
				dfact = nxk(itr)*njk(j,itr)/rhok(itr)
				Tfact = SQRT((mj(j)*kerg*Tx+ mx*kerg*Tk(itr))/(mx*mj(j))) * kerg*(Tx- Tk(itr))	! Tx-Tk gives correct sign
				xheat(itr) = xheat(itr)+ mfact* dfact* Tfact
		!		WRITE(10,*) itr, xheat(itr)
			ENDDO
		ENDDO
	ENDIF



	END SUBROUTINE calc_xheat


!!----------------------------
!!	calculate wimp number density for each cell
!!	assumes Nx has been calculated
!!----------------------------
	SUBROUTINE calc_nxk()
	USE const_def, only : pi, kerg ! Boltzmann's constant (erg K^-1)
	IMPLICIT NONE
	!INCLUDE 'wimp_vars.h'
	INTEGER :: itr
	DOUBLE PRECISION :: norm_integral, norm

	norm_integral = 0.D0
	DO itr=1,kmax!, 1, -1 ! integrate from r = 0 to Rstar
		norm_integral = norm_integral+ rk(itr+1)*rk(itr+1)* EXP(-mx*Vk(itr)/ kerg/Tx)* &
		(rk(itr)- rk(itr+1))
	ENDDO
	norm = Nx/ (4.D0*pi* norm_integral)

	nxk(kmax+1) = norm  ! this is central nx value since Vk(center) = 0
	DO itr = 1,kmax
		nxk(itr) = norm* EXP(-mx*Vk(itr)/ kerg/Tx)
	ENDDO

	END SUBROUTINE calc_nxk


!!----------------------------
!!	calculate number of captured wimps in current time step
!!	crate is ZH11 equation 1 with cboost subsuming rho and vbar factors
!!----------------------------
	FUNCTION calc_dNx()
	USE const_def, only : Msun ! solar mass (g)
	IMPLICIT NONE
	!INCLUDE 'wimp_vars.h'

	DOUBLE PRECISION :: calc_dNx, crate, Cfact

	IF (spindep) THEN
		Cfact = 5.D21* 5.D0/mxGeV ! s^-1
		crate = Cfact* cboost* sigmaxp/1.D-43* (vesc/6.18D7)**2* M_star/Msun
	ELSE
		Cfact = 7.D22 ! s^-1
		crate = Cfact* cboost* sigmaX/1.D-43* (vesc/6.18D7)**2* M_star/Msun
	ENDIF

	calc_dNx = crate* dttmp

	END FUNCTION calc_dNx



!!----------------------------
!!	calculate wimp temp using SP85 one-zone model.
!!	this is the root of SP85 equ 4.10
!!----------------------------
	FUNCTION calc_Tx()
	IMPLICIT NONE
	!INCLUDE 'wimp_vars.h'

	DOUBLE PRECISION :: Txhigh, Txlow, tol
	DOUBLE PRECISION :: Ttmp, calc_Tx
	PARAMETER ( tol = 1.D-4 )

	Txhigh = maxT
	Txlow = Txhigh/1.D2
	Ttmp = zbrent(emoment, Txhigh, Txlow, tol)

	calc_Tx = Ttmp
	END FUNCTION calc_Tx





!!----------------------------
!!	this is LHS of SP85 equ 4.10 which implicitly defines Tx
!!	called by zbrent() (which is called by calc_Tx())
!!	zbrent() finds Tx as the root of this equation
!!----------------------------
	FUNCTION emoment(Txtest)
	USE const_def, only : kerg ! Boltzmann's constant (erg K^-1)
	IMPLICIT NONE
	!INCLUDE 'wimp_vars.h'

	INTEGER :: itr, j
	DOUBLE PRECISION, INTENT(IN) :: Txtest
	DOUBLE PRECISION :: mpGeV, Tfact, efact, rfact, sum, emoment
	PARAMETER ( mpGeV=0.938272D0 ) ! Proton mass in GeV

	sum = 0.D0
	WRITE(*,*) 'emoment fnc, kmax =',kmax
	DO itr = kmax,1,-1 ! integrate from r=0 to r_star
		efact = EXP(-mx*Vk(itr)/ kerg/Txtest)
		rfact = rk(itr+1)*rk(itr+1)* (rk(itr)- rk(itr+1))
		IF (spindep) THEN
			Tfact = SQRT((mpGeV*Txtest+ mxGeV*Tk(itr))/(mxGeV*mpGeV))* (Tk(itr)-Txtest)
			sum = sum+ npk(itr)*Tfact*efact*rfact
		ELSE
			DO j = 1,numspecies
				Tfact = SQRT((mGeVj(j)*Txtest+ mxGeV*Tk(itr))/(mxGeV*mGeVj(j)))* (Tk(itr)-Txtest)
				sum = sum+ njk(j,itr)*Tfact*efact*rfact
			ENDDO
		ENDIF
	ENDDO

	emoment = sum
	END FUNCTION emoment


!!----------------------------
!!
!!----------------------------



!!----------------------------
!!
!!----------------------------


	END MODULE wimp_module
