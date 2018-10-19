!!!-------------------------------------------!!!
!!!	Troy Joseph Raen
!!!	June 2017
!!!
!!!	add-on module for MESA
!!!	calculates extra heat transported by WIMPS
!!!-------------------------------------------!!!
!!! controls:
!!! Nx = s% xtra1
!!!	cboost = s% X_CTRL(1)
!!!	spindep = s% X_LOGICAL_CTRL(1)  ! .true. = spin dependent; .false. = spin independent
!!!	extra history columns values = s% X_CTRL(2:6)


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
	INTEGER, INTENT(IN) :: id
	INTEGER, INTENT(OUT) :: ierr
	INTEGER :: itr
	TYPE (star_info), pointer :: s ! pointer to star type
	ierr=0
	CALL GET_STAR_PTR(id, s, ierr)
	IF ( ierr /= 0 ) RETURN

	cboost = s% X_CTRL(1)  ! boost in capture rate of WIMPs compared to the local capture rate near the Sun, \propto density/sigma_v
	spindep = s% X_LOGICAL_CTRL(1)
	CALL get_star_variables(id,ierr)
	CALL set_wimp_variables(id,ierr)
	CALL calc_xheat(Tx)

	DO itr = 1,kmax
		s% extra_heat(itr) = xheat(itr)
		! s% extra_heat(itr) = 1.0
		s% d_extra_heat_dlndm1(itr) = 0.D0
        s% d_extra_heat_dlnd00(itr) = d_xheat_dlnd00(itr)
		! s% d_extra_heat_dlnd00(itr) = 0.D0
        s% d_extra_heat_dlndp1(itr) = 0.D0
        s% d_extra_heat_dlnTm1(itr) = 0.D0
        s% d_extra_heat_dlnT00(itr) = d_xheat_dlnT00(itr)
		! s% d_extra_heat_dlnT00(itr) = 0.D0
        s% d_extra_heat_dlnTp1(itr) = 0.D0
        s% d_extra_heat_dlnR00(itr) = 0.D0
        s% d_extra_heat_dlnRp1(itr) = 0.D0
	ENDDO

	CALL store_hist(id,ierr)


	END SUBROUTINE wimp_energy_transport


!!----------------------------
!!	grab star variables from MESA
!!	calculate vesc and V(k)
!!----------------------------
	SUBROUTINE get_star_variables(id,ierr)
	use const_def, only : mp, Rsun, standard_cgrav, amu
	! proton mass (g), solar radius (cm), Grav const (g^-1 cm^3 s^-2), amu (g)
	IMPLICIT NONE
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

	DO itr = kmax,1, -1
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
				IF (itr .EQ. kmax) THEN
					mj(j) = chem_isos% W(chemj) * amu ! mass of element j (g)
					mGeVj(j) = mj(j)/gperGeV ! mass in GeV
					Aj(j) = chem_isos% Z_plus_N(chemj) ! mass number of element j
				ENDIF
				xajk(j,itr) = s% xa(j,itr) ! mass fraction of element j in cell k
				njk(j,itr) = xajk(j,itr)*rhok(itr)/mj(j) ! number fraction of element j in cell k
				IF (itr .EQ. kmax) THEN
!					WRITE(*,*) 'mod:  ',chem_isos% name(chemj), mGeVj(j), Aj(j), xajk(j,itr), njk(j,itr)
				ENDIF
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
		Vk(itr) = Vk(itr+1)+ 0.5D0*(gravk(itr)+ gravk(itr+1))* (rk(itr)- rk(itr+1)) ! cm^2/s^2
	ENDDO

	END SUBROUTINE get_star_variables




!!----------------------------
!!	set all WIMP properties
!!----------------------------
	SUBROUTINE set_wimp_variables(id,ierr)
	IMPLICIT NONE
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
!	cboost = s% X_CTRL(1)  ! boost in capture rate of WIMPs compared to the local capture rate near the Sun, \propto density/sigma_v
!	already set in main module
	IF (spindep) THEN
		sigmaxp = 1.D-37	! wimp-proton cross section, cm^2
	ELSE
		sigmaxp = 1.D-40
		DO j = 1,numspecies
			sigmaxj(j) = sigmaxp* (Aj(j)*mj(j)/mp)**2 * ((mx+mp)/(mx+mj(j)))**2
		ENDDO
	ENDIF

	! IF (cboost == 0.D0) THEN
	! 	Tx = 0.D0
	! ELSE
	Tx = calc_Tx(id,ierr)
	! ENDIF

!! in extras_finish_step (run_star_extras) s% xtra1 = s% xtra2
!! so wimps are not collected when step is not accepted
!	WRITE(*,*) 's% xtra1 = ', s% xtra1
	dNx = calc_dNx()
	Nx = (s% xtra1) + dNx
!	WRITE(*,*) 's% xtra1 = ', s% xtra1, 'dNx = ', dNx, 'Nx = ', Nx
	s% xtra2 = Nx
!	WRITE(*,*) 'mod:  Tx =',Tx, '  dNx =',dNx, '  Nx =',Nx
	CALL calc_nxk()

	END SUBROUTINE set_wimp_variables



!!----------------------------
!!	xheat is luminosity per unit mass (in cgs units) that wimps give up to baryons
!!	This is what needs to be given to the star program
!!	This is SP85 equ 4.9 divided by rho(k)
!!----------------------------
	SUBROUTINE calc_xheat(Tx_given)
	USE const_def, only : pi, mp, kerg ! Boltzmann's constant (erg K^-1)
	IMPLICIT NONE
	INTEGER :: itr, j
	DOUBLE PRECISION :: mfact, dfact, Tfact, xheat_j, Tx_given

!	LOGICAL :: ISOPEN

!	INQUIRE(FILE='xheat.Tx_givent', OPENED=ISOPEN)
!	IF (.NOT. ISOPEN) THEN
!		OPEN(FILE='xheat.Tx_givent', UNIT=10)
!	ENDIF

	IF (spindep) THEN
		mfact = 8.D0*SQRT(2.D0/pi)* sigmaxp* mx*mp/((mx+mp)**2) ! this is common to all cells

		DO itr = 1,kmax
			dfact = nxk(itr)*npk(itr)/rhok(itr)
			Tfact = SQRT((mp*kerg*Tx_given+ mx*kerg*Tk(itr))/(mx*mp)) * kerg*(Tx_given- Tk(itr))	! Tx_given-Tk gives correct sign
			xheat(itr) = mfact* dfact* Tfact

			! partial drvs for other_energy_implicit. all those not calculated here are zero.
			d_xheat_dlnd00(itr) = -xheat(itr)
			Tfact = mx*kerg/2.D0/(mp*kerg*Tx_given+ mx*kerg*Tk(itr)) - 1.D0/(Tx_given-Tk(itr))
			d_xheat_dlnT00(itr) = xheat(itr)*Tk(itr)*Tfact
	!		WRITE(10,*) itr, xheat(itr)
		ENDDO
	!	DO itr = 1,kmax
	!		WRITE(10,"(F10.5)",advance="no") xheat(itr)
	!	ENDDO

	ELSE
		DO itr = 1,kmax
			xheat(itr) = 0.D0
			d_xheat_dlnd00(itr) = 0.D0
			d_xheat_dlnT00(itr) = 0.D0
			DO j = 1,numspecies
				mfact = 8.D0*SQRT(2.D0/pi)* sigmaxj(j)* mx*mj(j)/((mx+mj(j))**2)
				dfact = nxk(itr)*njk(j,itr)/rhok(itr)
				Tfact = SQRT((mj(j)*kerg*Tx_given+ mx*kerg*Tk(itr))/(mx*mj(j))) * kerg*(Tx_given- Tk(itr))	! Tx_given-Tk gives correct sign
				xheat_j = mfact* dfact* Tfact
				xheat(itr) = xheat(itr)+ xheat_j
				! partial drvs for other_energy_implicit. all those not calculated here are zero.
				d_xheat_dlnd00(itr) = d_xheat_dlnd00(itr) - xheat_j
				Tfact = mx*kerg/2.D0/(mj(j)*kerg*Tx_given+ mx*kerg*Tk(itr)) - 1.D0/(Tx_given-Tk(itr))
				d_xheat_dlnT00(itr) = d_xheat_dlnT00(itr) + xheat_j*Tk(itr)*Tfact
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
	INTEGER :: itr
	DOUBLE PRECISION :: norm_integral, norm

	norm_integral = 0.D0
	DO itr=1,kmax!, 1, -1 ! integrate from r = 0 to Rstar
		norm_integral = norm_integral+ rk(itr+1)*rk(itr+1)* &
		EXP(-mx*Vk(itr)/ kerg/Tx)* (rk(itr)- rk(itr+1))
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
	DOUBLE PRECISION :: calc_dNx, crate, Cfact

	IF (spindep) THEN
		Cfact = 5.D21* 5.D0/mxGeV ! s^-1
		crate = Cfact* cboost* sigmaxp/1.D-43* (vesc/6.18D7)**2* M_star/Msun
	ELSE
		Cfact = 7.D22 ! s^-1
		crate = Cfact* cboost* sigmaxp/1.D-43* (vesc/6.18D7)**2* M_star/Msun
	ENDIF

	calc_dNx = crate* dttmp

	END FUNCTION calc_dNx



!!----------------------------
!!	calculate wimp temp using SP85 one-zone model.
!!	this is the root of SP85 equ 4.10
!!----------------------------
	FUNCTION calc_Tx(id,ierr)
	USE nrutil, ONLY : nrerror
	USE const_def, only : Lsun
	IMPLICIT NONE
	INTEGER, INTENT(IN) :: id
	INTEGER, INTENT(OUT) :: ierr
	INTEGER :: k
	DOUBLE PRECISION :: Txhigh, Txlow, tol
	DOUBLE PRECISION :: Ttmp, calc_Tx, xL, Lnuc
	DOUBLE PRECISION :: Tarray(4)
	INTEGER :: tries, model_err=-1
	LOGICAL :: Tflag=.FALSE.
	! PARAMETER ( tol = 1.D-4 )
	TYPE (star_info), pointer :: s ! pointer to star type
	ierr=0
	CALL GET_STAR_PTR(id, s, ierr)
	IF ( ierr /= 0 ) RETURN

	tol = s% X_CTRL(7)

	IF ((model_err.EQ. s% model_number) .AND. (.NOT.Tflag)) call nrerror('Txlow > Txhigh or root must be bracketed')
	Txhigh = maxT*1.1
	Txlow = maxT/25.0
!	Ttmp = 10.0D0

	tries=0
	Tflag=.FALSE.
	! WRITE(*,*) '#****# BEGIN #****# '
	! WRITE(*,*) 'Model = ', s% model_number
	! WRITE(*,*) 'Tflag = ', Tflag
	DO WHILE ( .NOT. Tflag )
		tries = tries+1

		! IF (Txlow.GT.maxT) THEN !call nrerror('Txlow > Txhigh')
		! 	Ttmp=10.**(s% log_center_temperature)
		! 	model_err= s% model_number +1
		! 	WRITE(*,*) 'Txlow>maxT. problem model ', s% model_number
		! 	WRITE(*,*) 'tries=', tries, 'Txhigh=', Txhigh, 'Txlow=', Txlow
		! 	EXIT
		! ENDIF

		Tarray = zbrent(emoment, Txhigh, Txlow, tol) ! returns -1 if gets root must be bracketed error
		Ttmp = Tarray(1)
		! WRITE(*,*) 'Ttmp = ', Ttmp
		IF (Txlow.GT.maxT) THEN ! treat as root must be bracketed error. Tx close to Tmax and slope is shallow (most likely)
			WRITE(*,*) 'Txlow > maxT. Treating as root must be bracketed error. Model =', s% model_number
			Ttmp = -1.0
		ENDIF

		IF (Ttmp.GT.0.0) THEN
			Tflag = is_slope_negative(Ttmp)
			! WRITE(*,*) 'Tflag = ', Tflag
			Txlow = 1.05*Txlow
!			WRITE(*,*) 'tries=', tries, 'Txhigh=', Txhigh, 'Txlow=', Txlow, 'Ttmp=', Ttmp
		ELSE IF (tries.EQ.1) THEN ! expand the range and try again
			WRITE(*,*) 'root must be bracketed. Txhigh=', Txhigh, 'Txlow=', Txlow, 'tries=', tries, 'Model =', s% model_number
			Txlow = Txlow/2.0
			Txhigh = Txhigh*1.25
		ELSE ! go back a step, find root, make sure slope is negative
			WRITE(*,*) 'root must be bracketed. Txhigh=', Txhigh, 'Txlow=', Txlow, 'tries=', tries, 'Model =', s% model_number
			Txlow = Txlow/1.05
			Tarray = zbrent(emoment, Txhigh, Txlow, tol)
			Ttmp = Tarray(1)
			Tflag = is_slope_negative(Ttmp)
			! WRITE(*,*) 'Tflag = ', Tflag
			IF (.NOT.Tflag) THEN
				Ttmp= 10.**(s% log_center_temperature)
				WRITE(*,*) '**** Tx set to center_T ****', Ttmp, 'problem model ', s% model_number
				model_err= s% model_number +1
				EXIT ! terminate the DO WHILE loop
			ENDIF

		ENDIF

	ENDDO

	! WRITE(*,*) "ZBRENT---***--- Tx1, emom1, Tx2, emom2", Tarray(1), Tarray(2), Tarray(3), Tarray(4)
	! WRITE(*,*) '#****# END #****# '

	!!!!!!!!
	! check that L_extra / L_nuc < 1
	! else approximate emoment root function
	! with straight line to find better Tx
	CALL calc_xheat(Ttmp)
	xL = 0.0
	DO k = 1,kmax
		xL = xL + xheat(k)* s% dq(k)* s% xmstar ! (ergs/gm/sec)*gm = ergs/sec
	ENDDO
	xL = xL/Lsun
	Lnuc = s% power_nuc_burn ! Lsun
	s% xtra4 = Lnuc
	s% xtra5 = xL
	! IF (( ABS(xL/ Lnuc).GT.0.1 ) .AND. ( Lnuc.GT.0.1)) THEN !
	! 	Ttmp = linear_root(Tarray)
	! 	WRITE(*,*) "ZBRENT---***---"
	! 	WRITE(*,*) "ZBRENT---***--- model number = ", s% model_number
	! 	WRITE(*,*) "ZBRENT---***--- xL/ Lnuc OLD = ", xL/ Lnuc,  "---***---"
	! 	CALL calc_xheat(Ttmp)
	! 	xL = 0.0
	! 	DO k = 1,kmax
	! 		xL = xL + xheat(k)* s% dq(k)* s% xmstar ! (ergs/gm/sec)*gm = ergs/sec
	! 	ENDDO
	! 	xL = xL/Lsun
	! 	WRITE(*,*) "ZBRENT---***--- xL/ Lnuc NEW = ", xL/ Lnuc,  "---***---"
	! ENDIF
	s% xtra6 = xL/Lnuc
	!!!!!!!!


!!!!!!!!
	CALL test_routine(id,ierr,Tarray)
!!!!!!!!


	calc_Tx = Ttmp
	END FUNCTION calc_Tx


!!!!!!!!
! already have [ Tx1, emom1, Tx2, emom2 ] from wimp_num root finding.
! find linear_root Tx and emom with this Tx
! calc xenergy for all these Tx's
!!!!!!!!
	SUBROUTINE test_routine(id,ierr,Tarray)
		INTEGER, INTENT(IN) :: id
		INTEGER, INTENT(OUT) :: ierr
		INTEGER :: i, j
		DOUBLE PRECISION :: Tarray(4), matrx(3,3)
		! matrx = 	[ Tx1, emom(Tx1), xEnergy(Tx1) ]
		!			[ Tx2, emom(Tx2), xEnergy(Tx2) ]
		!			[ Tx_linapprox, emom(Tx_linapprox), xEnergy(Tx_linapprox) ]
		TYPE (star_info), pointer :: s ! pointer to star type
		ierr=0
		CALL GET_STAR_PTR(id, s, ierr)
		IF ( ierr /= 0 ) RETURN

		matrx(1,1) = Tarray(1)
		matrx(1,2) = Tarray(2)
		matrx(2,1) = Tarray(3)
		matrx(2,2) = Tarray(4)
		matrx(3,1) = linear_root(Tarray)
		matrx(3,2) = emoment(matrx(3,1))

		DO i = 1,3
			CALL calc_xheat(matrx(i,1))
			matrx(i,3) = calc_xenergy(id,ierr)
		ENDDO

		OPEN(UNIT=9, FILE="/home/tjr63/mesaruns/LOGS/matrx.data", STATUS="NEW", ACTION="WRITE")
		WRITE(UNIT=9, FMT="(3F15.2)") ((matrx(i,j), i = 1, 3), j = 1, 3)
		CLOSE(UNIT=9)

		STOP

	END SUBROUTINE test_routine

!!! ONLY USED FOR test_routine:
	FUNCTION calc_xenergy(id, ierr)
		integer, intent(in) :: id, ierr
		real(dp) :: xe, calc_xenergy
		integer :: k
		type (star_info), pointer :: s
		ierr = 0
		call GET_STAR_PTR(id, s, ierr)
		if (ierr /= 0) return

		xe = 0.d0
		DO k = 1, s% nz
			xe = xe + s% extra_heat(k)* s% dm(k)* s% dt
		ENDDO

		calc_xenergy = xe ! ergs
	END FUNCTION calc_xenergy
!!! ONLY USED FOR test_routine


!!----------------------------
!!	If extra energy is "too high", approximate emoment function with a
!!	straight line and return a root closer to actual zero
	FUNCTION linear_root(Tarray)
		DOUBLE PRECISION :: x1, x2, y1, y2
		DOUBLE PRECISION :: m, linear_root
		DOUBLE PRECISION :: Tarray(4)

		x1 = Tarray(1)
		y1 = Tarray(2)
		x2 = Tarray(3)
		y2 = Tarray(4)
		m = (y2-y1)/(x2-x1)
		linear_root = x1 - y1/m ! Tx

		WRITE(*,*) "ZBRENT---***--- slope = ", m
		WRITE(*,*) "ZBRENT---***--- Tx OLD = ", x1
		WRITE(*,*) "ZBRENT---***--- emoment OLD = ", y1
		WRITE(*,*) "ZBRENT---***--- Tx NEW = ", linear_root
		WRITE(*,*) "ZBRENT---***--- emoment NEW = ", emoment(linear_root)
		WRITE(*,*) "ZBRENT---***---"

	END FUNCTION linear_root


!!----------------------------
!!	To avoid finding wrong (lower) root and inducing Tx oscillations
!!	calculate slope of Ttmp
!!	lower root has slope ~0, correct slope is usually very steep
!!	if abs(slope) < 10, return .FALSE.
	FUNCTION is_slope_steep(Tx_given)
		IMPLICIT NONE
		DOUBLE PRECISION, INTENT(IN) :: Tx_given
		DOUBLE PRECISION :: Tl, Tx_int, Tl_int, slope
		LOGICAL :: is_slope_steep

		Tx_int = emoment(Tx_given)
		Tl = 0.99*Tx
		Tl_int = emoment(Tl)
		slope = ABS((Tx_int-Tl_int)/(Tx_given-Tl))
		! WRITE(*,*) 'slope=',slope

		IF (slope.GT.10.0) THEN
			is_slope_steep=.TRUE.
			WRITE(*,*) 'is_slope_steep returns true. slope=',slope
		ELSE
			is_slope_steep=.FALSE.
			WRITE(*,*) 'is_slope_steep returns false. slope=',slope
		ENDIF

	END FUNCTION is_slope_steep
!!	if slope < 0, return .TRUE.
!!----------------------------
	FUNCTION is_slope_negative(Tx_given)
		IMPLICIT NONE
		DOUBLE PRECISION, INTENT(IN) :: Tx_given
		DOUBLE PRECISION :: Tl, Tx_int, Tl_int, slope
		LOGICAL :: is_slope_negative

		Tx_int = emoment(Tx_given)
		Tl = 0.999*Tx_given ! use narrower range to avoid function maximum
		Tl_int = emoment(Tl)
		slope = (Tx_int-Tl_int)/(Tx_given-Tl)

		IF (slope.LT.0.0) THEN
			is_slope_negative=.TRUE.
			! WRITE(*,*) 'is_slope_negative returns true. slope=',slope
		ELSE
			is_slope_negative=.FALSE.
			WRITE(*,*) 'is_slope_negative returns false. slope=',slope
		ENDIF

	END FUNCTION is_slope_negative



!!----------------------------
!!	this is LHS of SP85 equ 4.10 which implicitly defines Tx
!!	called by zbrent() (which is called by calc_Tx())
!!	zbrent() finds Tx as the root of this equation
!!----------------------------
	FUNCTION emoment(Txtest)
	USE const_def, only : Rsun, kerg ! Boltzmann's constant (erg K^-1)
	IMPLICIT NONE
	INTEGER :: itr, j
	DOUBLE PRECISION, INTENT(IN) :: Txtest
	DOUBLE PRECISION :: mpGeV, Tfact, efact, rfact, mjfact, sum, emoment
	DOUBLE PRECISION :: Tnorm, nnorm, Rnorm, m, npbar, Txbar, Tbar, rbar, drbar
	PARAMETER ( mpGeV=0.938272D0 ) ! Proton mass in GeV

! !!!! normalized
! 	! normalization constants
! 	Tnorm = 1.D7 ! K
! 	nnorm = 1.D25 ! dimensionless
! 	Rnorm = Rsun ! cm
! 	! normalized variables
! 	m = mxGeV / mpGeV
! 	Txbar = Txtest / Tnorm
!
! 	sum = 0.D0
! 	DO itr = kmax,1,-1 ! integrate from r=0 to r_star
! 		! normalized variables
! 		npbar = npk(itr) / nnorm
! 		Tbar = Tk(itr) / Tnorm
! 		rbar = rk(itr+1) / Rnorm
! 		drbar = (rk(itr)- rk(itr+1)) / Rnorm
!
! 		rfact = rbar*rbar*drbar
! 		efact = EXP(-mx*Vk(itr)/ kerg/Txtest)
! 		IF (spindep) THEN
! 			Tfact = SQRT(Txbar+ m*Tbar)* (Tbar - Txbar)
! 			sum = sum+ npbar*Tfact*efact*rfact
! !!!! STILL NEED SPIN INDEPENDENT NORMALIZED
! 		ENDIF
! 	ENDDO
! !!!! end normalized

!!!! non-normalized
	sum = 0.D0
	DO itr = kmax,1,-1 ! integrate from r=0 to r_star
		rfact = rk(itr+1)*rk(itr+1)* (rk(itr)- rk(itr+1))
		efact = EXP(-mx*Vk(itr)/ kerg/Txtest)
		IF (spindep) THEN
			Tfact = SQRT((mpGeV*Txtest+ mxGeV*Tk(itr))/(mxGeV*mpGeV))* (Tk(itr)-Txtest)
			sum = sum+ npk(itr)*Tfact*efact*rfact
		ELSE
			DO j = 1,numspecies
				Tfact = SQRT((mGeVj(j)*Txtest+ mxGeV*Tk(itr))/(mxGeV*mGeVj(j)))* (Tk(itr)-Txtest)
				mjfact = Aj(j)*Aj(j)* (mGeVj(j)/mpGeV)**3 *(mxGeV+mpGeV)**2 &
				*mxGeV*mpGeV/ (mxGeV+mGeVj(j))**4
				sum = sum+ njk(j,itr)*Tfact*efact*rfact*mjfact
			ENDDO
		ENDIF
	ENDDO
!!!! end non-normalized

	emoment = sum
	END FUNCTION emoment

!!----------------------------
!!	store data so it can be written to history.data
!!	THIS INFO MUST MATCH subroutine data_for_extra_history_columns
!!	IN run_star_extras !!!
!!----------------------------
	SUBROUTINE store_hist(id,ierr)
		IMPLICIT NONE
		INTEGER, INTENT(IN) :: id
		INTEGER, INTENT(OUT) :: ierr
		INTEGER :: j, k, idx
		TYPE (star_info), pointer :: s ! pointer to star type
		ierr=0
		CALL GET_STAR_PTR(id, s, ierr)
		IF ( ierr /= 0 ) RETURN

		! store info to write to extra_history_columns
		! indicies offset by one since cboost = s% X_CTRL(1)

		s% X_CTRL(2) = Tx ! names(1) = 'wimp_temp'
		s% X_CTRL(3) = Nx ! names(2) = 'Nx_total'
		s% X_CTRL(4) = nxk((s% nz)+1) ! names(3) = 'center_nx'
		s% X_CTRL(5) = npk((s% nz)+1) ! names(4) = 'center_np'
		s% X_CTRL(6) = emoment(Tx) ! names(5) = 'Tx_emoment'

		DO k = 1,kmax
			s% xtra1_array(k) = nxk(k)
			s% xtra2_array(k) = npk(k)
			s% xtra3_array(k) = Vk(k)
		ENDDO



	END SUBROUTINE store_hist


	END MODULE wimp_module
