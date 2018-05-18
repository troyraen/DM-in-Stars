
	INTEGER, PARAMETER :: maxspecies=11
	INTEGER, PARAMETER :: maxcells=5000	!// if you change this, change num_x_ctrls in inlist
	DOUBLE PRECISION, PARAMETER ::  gperGeV= 1.78266D-24 ! grams to GeV/c^2 conversion

	LOGICAL :: spindep
	INTEGER :: chemj, numspecies, kmax  !// zones go from k= 1 to kmax. kmax+1 is star center

!// 	! wimp variables
	DOUBLE PRECISION mx, mxGeV, sigmaxp, cboost, Tx, Nx


!//	! star variables
	DOUBLE PRECISION dttmp, maxT, vesc, M_star, R_star, Age_star
!//! defined at cell center:
	DOUBLE PRECISION Xk(1:maxcells)
	DOUBLE PRECISION Tk(1:maxcells)
	DOUBLE PRECISION rhok(1:maxcells)
	DOUBLE PRECISION npk(1:maxcells)
	DOUBLE PRECISION nxk(1:maxcells)
	DOUBLE PRECISION xheat(1:maxcells)
	DOUBLE PRECISION d_xheat_dlnd00(1:maxcells)
	DOUBLE PRECISION d_xheat_dlnT00(1:maxcells)
!//! defined at cell outer face:
	DOUBLE PRECISION rk(1:maxcells)
	DOUBLE PRECISION gravk(1:maxcells)
!//! face values averaged to define Vk at cell center:
	DOUBLE PRECISION Vk(1:maxcells)

	REAL(DP), DIMENSION(1:maxspecies,1:maxcells) :: xajk, njk
	REAL(DP), DIMENSION(1:maxspecies) :: mj, mGeVj, sigmaxj
	INTEGER, DIMENSION(1:maxspecies) :: Aj
