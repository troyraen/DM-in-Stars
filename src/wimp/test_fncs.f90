!!!-------------------------------------------!!!
!!!	Troy Joseph Raen
!!!
!!!-------------------------------------------!!!
!!! controls:
!!! Nx = s% xtra1
!!!	cboost = s% X_CTRL(1)
!!!	spindep = s% X_LOGICAL_CTRL(1)  ! .true. = spin dependent; .false. = spin independent
!!!	extra history columns values = s% X_CTRL(2:6)



MODULE test_fncs

! use const_def
use wimp_module
! use chem_def
! use wimp_num
IMPLICIT NONE
INCLUDE 'wimp_vars.h'

CONTAINS


!!!!!!!!
! have Tx_array = [ Tx1, emom1, Tx2, emom2 ] from wimp_num root finding.
! find linear_root Tx and emom with this Tx
! calc xenergy for all these Tx's
! write to separate file for plotting
!!!!!!!!
	SUBROUTINE energy_plots(id,ierr)
        use wimp_module
		INTEGER, INTENT(IN) :: id
		INTEGER, INTENT(OUT) :: ierr
		INTEGER :: i, j, k
		DOUBLE PRECISION :: matrx(3,853)
		! matrx = 	[ Tx1, emom(Tx1), xEnergy(Tx1) xheat(Tx1) for each zone ]
		!			[ Tx2, emom(Tx2), xEnergy(Tx2) xheat(Tx2) for each zone ]
		!			[ Tx_linapprox, emom(Tx_linapprox), xEnergy(Tx_linapprox) xheat(Tx_linapprox) for each zone ]
		TYPE (star_info), pointer :: s ! pointer to star type
		ierr=0
		CALL GET_STAR_PTR(id, s, ierr)
		IF ( ierr /= 0 ) RETURN

		matrx(1,1) = Tx_array(1)
		matrx(1,2) = Tx_array(2)
		matrx(2,1) = Tx_array(3)
		matrx(2,2) = Tx_array(4)
		matrx(3,1) = linear_root(Tx_array)
		matrx(3,2) = emoment(matrx(3,1))

		DO i = 1,3
			CALL calc_xheat(matrx(i,1))
			matrx(i,3) = calc_xenrgy(id,ierr)
            DO k = 1,kmax
                matrx(i,k+3) = xheat(k)
            ENDDO
            DO k = kmax+1,850
                matrx(i,k+3) = 0.0D0
            ENDDO
		ENDDO

		OPEN(UNIT=9, FILE="/home/tjr63/mesaruns/LOGS/matrx.data", STATUS="NEW", ACTION="WRITE")
		WRITE(UNIT=9, FMT="(3F15.2)") ((matrx(i,j), i = 1, 3), j = 1, 853)
		CLOSE(UNIT=9)

		STOP

	END SUBROUTINE energy_plots


!!! ONLY USED FOR test_routine:
	FUNCTION calc_xenrgy(id, ierr)
        use star_def
		INTEGER, INTENT(IN) :: id
		INTEGER, INTENT(OUT) :: ierr
		real(dp) :: xe, calc_xenrgy
		integer :: k
		type (star_info), pointer :: s
		ierr = 0
		call GET_STAR_PTR(id, s, ierr)
		if (ierr /= 0) return

		xe = 0.d0
		DO k = 1, kmax
			xe = xe + xheat(k)* s% dm(k)* s% dt
		ENDDO

		calc_xenrgy = xe ! ergs
	END FUNCTION calc_xenrgy
!!! ONLY USED FOR test_routine



!!----------------------------
!!	this is LHS of SP85 equ 4.10 which implicitly defines Tx
!!	called by emoment function if inlist specifies it should be normalized
!!----------------------------
	FUNCTION emom_normalized(Txtest)
		USE const_def, only : Rsun, kerg ! Boltzmann's constant (erg K^-1)
		IMPLICIT NONE
		INTEGER :: itr, j
		DOUBLE PRECISION, INTENT(IN) :: Txtest
		DOUBLE PRECISION :: mpGeV, Tfact, efact, rfact, mjfact, sum, emom_normalized
		DOUBLE PRECISION :: Tnorm, nnorm, Rnorm, m, npbar, Txbar, Tbar, rbar, drbar
		PARAMETER ( mpGeV=0.938272D0 ) ! Proton mass in GeV

	!!!! normalized
		! normalization constants
		Tnorm = 1.D7 ! K
		nnorm = 1.D25 ! dimensionless
		Rnorm = Rsun ! cm
		! normalized variables
		m = mxGeV / mpGeV
		Txbar = Txtest / Tnorm

		sum = 0.D0
		DO itr = kmax,1,-1 ! integrate from r=0 to r_star
			! normalized variables
			npbar = npk(itr) / nnorm
			Tbar = Tk(itr) / Tnorm
			rbar = rk(itr+1) / Rnorm
			drbar = (rk(itr)- rk(itr+1)) / Rnorm

			rfact = rbar*rbar*drbar
			efact = EXP(-mx*Vk(itr)/ kerg/Txtest)
			IF (spindep) THEN
				Tfact = SQRT(Txbar+ m*Tbar)* (Tbar - Txbar)
				sum = sum+ npbar*Tfact*efact*rfact
	!!!! STILL NEED SPIN INDEPENDENT NORMALIZED
			ENDIF
		ENDDO
	!!!! end normalized



		emom_normalized = sum

	END FUNCTION emom_normalized


END MODULE test_fncs
