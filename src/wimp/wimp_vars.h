
	INTEGER maxcells, kmax  ! zones go from k= 1 to kmax. kmax+1 is star center
	PARAMETER ( maxcells=10000 )
	INTEGER maxspecies, numspecies
	PARAMETER ( maxspecies=300)


! 	! wimp variables
	DOUBLE PRECISION mx, mxGeV, sigmaxp, cboost, Tx, Nx
	LOGICAL spindep


!	! star variables
	DOUBLE PRECISION dttmp, maxT, vesc, M_star, R_star, Age_star
!! defined at cell center:
	DOUBLE PRECISION Xk(1:maxcells)
	DOUBLE PRECISION xajk(1:maxspecies, 1:maxcells)
	DOUBLE PRECISION Tk(1:maxcells)
	DOUBLE PRECISION rhok(1:maxcells)
	DOUBLE PRECISION npk(1:maxcells)
	DOUBLE PRECISION nNk(1:maxspecies, 1:maxcells)
	DOUBLE PRECISION nxk(1:maxcells)
	DOUBLE PRECISION xheat(1:maxcells)
!! defined at cell outer face:
	DOUBLE PRECISION rk(1:maxcells)
	DOUBLE PRECISION gravk(1:maxcells)
!! face values averaged to define Vk at cell center:
	DOUBLE PRECISION Vk(1:maxcells)

	COMMON /WIMP_MOD_GLOBALS/ mx, mxGeV, sigmaxp, cboost, Tx, Nx, dttmp, maxT, vesc, M_star, R_star, Age_star
	COMMON /WIMP_MOD_GLOBALS/ Xk, Tk, rhok, npk, nxk, xheat, rk, gravk, Vk
	COMMON /WIMP_MOD_GLOBALS/ kmax, spindep, maxspecies, numspecies, xajk, nNk
