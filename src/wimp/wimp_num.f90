!!----------------------------
!!	returns the root of the function func
!!	taken from Numerical Recipes
!!----------------------------
	MODULE wimp_num

	contains


	FUNCTION zbrent(func,x1,x2,tol)
	USE nrtype; USE nrutil, ONLY : nrerror
	IMPLICIT NONE
	REAL(DP), INTENT(IN) :: x1,x2,tol
	REAL(DP), DIMENSION(4) :: zbrent
	INTERFACE
		FUNCTION func(x)
		USE nrtype
		IMPLICIT NONE
		REAL(DP), INTENT(IN) :: x
		REAL(DP) :: func
		END FUNCTION func
	END INTERFACE
	INTEGER(I4B), PARAMETER :: ITMAX=100
	REAL(DP), PARAMETER :: EPS=epsilon(x1)
	INTEGER(I4B) :: iter
	REAL(DP) :: a,b,c,d,e,fa,fb,fc,p,q,r,s,tol1,xm
	a=x1
	b=x2
	fa=func(a)
	fb=func(b)
	if ((fa > 0.0 .and. fb > 0.0) .or. (fa < 0.0 .and. fb < 0.0)) THEN
!		call nrerror('root must be bracketed for zbrent')
!		instead of exiting run, increase lower limit and try again
		zbrent = [-1.0D0, 0.0D0, 0.0D0, 0.0D0]
		RETURN
	ENDIF
	c=b
	fc=fb
	do iter=1,ITMAX
		if ((fb > 0.0 .and. fc > 0.0) .or. (fb < 0.0 .and. fc < 0.0)) then
			c=a
			fc=fa
			d=b-a
			e=d
		end if
		if (abs(fc) < abs(fb)) then
			a=b
			b=c
			c=a
			fa=fb
			fb=fc
			fc=fa
		end if
		tol1=2.0_dp*EPS*abs(b)+0.5_dp*tol
		xm=0.5_dp*(c-b)
		if (abs(xm) <= tol1 .or. fb == 0.0) then
			! Root (b) has been found.
			! Return this root plus the root from the other interval
			! (and both function evaluations).
			! This will be a or c depending on the signs of fa, fb, and fc.
			! The second root will be used to approximate the emoment
			! equation with a straight line so that a more accurate
			! root can be found when the extra energy is "too high".
			IF ( SIGN(1.0D0,fb) .EQ. -SIGN(1.0D0,fa) ) THEN
				zbrent = [b, fb, a, fa]
			ELSE IF ( SIGN(1.0D0,fb) .EQ. -SIGN(1.0D0,fc) ) THEN
				zbrent = [b, fb, c, fc]
			ELSE
				WRITE(*,*) "---***--- ALL ZBRENT ROOTS HAVE THE SAME SIGN ---***---"
				IF ( ABS(fa) .LT. ABS(fc) ) THEN
					zbrent = [b, fb, a, fa]
				ELSE
					zbrent = [b, fb, c, fc]
				END IF
			END IF
			! zbrent=b
			RETURN
		end if
		if (abs(e) >= tol1 .and. abs(fa) > abs(fb)) then
			s=fb/fa
			if (a == c) then
				p=2.0_dp*xm*s
				q=1.0_dp-s
			else
				q=fa/fc
				r=fb/fc
				p=s*(2.0_dp*xm*q*(q-r)-(b-a)*(r-1.0_dp))
				q=(q-1.0_dp)*(r-1.0_dp)*(s-1.0_dp)
			end if
			if (p > 0.0) q=-q
			p=abs(p)
			if (2.0_dp*p  <  min(3.0_dp*xm*q-abs(tol1*q),abs(e*q))) then
				e=d
				d=p/q
			else
				d=xm
				e=d
			end if
		else
			d=xm
			e=d
		end if
		a=b
		fa=fb
		b=b+merge(d,sign(tol1,xm), abs(d) > tol1 )
		fb=func(b)
	end do
	call nrerror('zbrent: exceeded maximum iterations')
	zbrent=[b, 0.0D0, 0.0D0, 0.0D0]
	END FUNCTION zbrent

	END MODULE wimp_num
