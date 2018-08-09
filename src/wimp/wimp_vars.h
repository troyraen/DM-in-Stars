
	INTEGER, PARAMETER :: maxspecies=11
	INTEGER, PARAMETER :: maxcells=5000
	REAL*4, PARAMETER ::  gperGeV= 1.78266D-24 ! grams to GeV/c^2 conversion

	LOGICAL :: spindep
	INTEGER :: chemj, numspecies, kmax  !// zones go from k= 1 to kmax. kmax+1 is star center

!// 	! wimp variables
	REAL*4 mx, mxGeV, sigmaxp, cboost, Tx, Nx


!//	! star variables
	REAL*4 dttmp, maxT, vesc, M_star, R_star, Age_star
!//! defined at cell center:
	REAL*4 Xk(1:maxcells)
	REAL*4 Tk(1:maxcells)
	REAL*4 rhok(1:maxcells)
	REAL*4 npk(1:maxcells)
	REAL*4 nxk(1:maxcells)
	REAL*4 xheat(1:maxcells)
	REAL*4 d_xheat_dlnd00(1:maxcells)
	REAL*4 d_xheat_dlnT00(1:maxcells)
!//! defined at cell outer face:
	REAL*4 rk(1:maxcells)
	REAL*4 gravk(1:maxcells)
!//! face values averaged to define Vk at cell center:
	REAL*4 Vk(1:maxcells)

	REAL(DP), DIMENSION(1:maxspecies,1:maxcells) :: xajk, njk
	REAL(DP), DIMENSION(1:maxspecies) :: mj, mGeVj, sigmaxj
	INTEGER, DIMENSION(1:maxspecies) :: Aj
