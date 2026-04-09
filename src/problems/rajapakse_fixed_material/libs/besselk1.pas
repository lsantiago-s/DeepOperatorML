unit besselk1;
interface
procedure kzeone(x,y: double; var re0, im0, re1, im1: double);

implementation
procedure kzeone(x,y: double; var re0, im0, re1, im1: double);


{C     ALGORITHM 484 COLLECTED ALGORITHMS FROM ACM.
C     ALGORITHM APPEARED IN COMM. ACM, VOL. 17, NO. 09,
C     P. 524.
      SUBROUTINE KZEONE(X, Y, RE0, IM0, RE1, IM1)                       KZE  10
C THE VARIABLES X AND Y ARE THE REAL AND IMAGINARY PARTS OF
C THE ARGUMENT OF THE FIRST TWO MODIFIED BESSEL FUNCTIONS
C OF THE SECOND KIND,K0 AND K1.  RE0,IM0,RE1 AND IM1 GIVE
C THE REAL AND IMAGINARY PARTS OF EXP(X)*K0 AND EXP(X)*K1,
C RESPECTIVELY.  ALTHOUGH THE REAL NOTATION USED IN THIS
C SUBROUTINE MAY SEEM INELEGANT WHEN COMPARED WITH THE
C COMPLEX NOTATION THAT FORTRAN ALLOWS, THIS VERSION RUNS
C ABOUT 30 PERCENT FASTER THAN ONE WRITTEN USING COMPLEX
C VARIABLES.}
var
{      DOUBLE PRECISION X, Y, X2, Y2, RE0, IM0, RE1, IM1,
     * R1, R2, T1, T2, P1, P2, RTERM, ITERM, EXSQ(8), TSQ(8)}
x2, y2, r1, r2, t1, t2, p1, p2, rterm, iterm: double;
n, l, m, k: integer;

const
tsq: array[1..8] of double =(
     0.0E0, 3.19303633920635E-1,
     1.29075862295915E0, 2.95837445869665E0,
     5.40903159724444E0, 8.80407957805676E0,
     1.34685357432515E1, 2.02499163658709E1);
     
exsq: array[1..8] of double =(
     0.5641003087264E0, 0.4120286874989E0,
     0.1584889157959E0, 0.3078003387255E-1,
     0.2778068842913E-2, 0.1000044412325E-3,
     0.1059115547711E-5, 0.1522475804254E-8);
     
     
{C THE ARRAYS TSQ AND EXSQ CONTAIN THE SQUARE OF THE
C ABSCISSAS AND THE WEIGHT FACTORS USED IN THE GAUSS-
C HERMITE QUADRATURE.}

begin
{      R2 = X*X + Y*Y
      IF (X.GT.0.0D0 .OR. R2.NE.0.0D0) GO TO 10
      WRITE (6,99999)
      RETURN}
r2:= x*x+y*y;
if (x<=0.0) and (r2=0.0) then exit;

{   10 IF (R2.GE.1.96D2) GO TO 50
      IF (R2.GE.1.849D1) GO TO 30}
if r2<1.849E1 then
   begin
{C THIS SECTION CALCULATES THE FUNCTIONS USING THE SERIES
C EXPANSIONS
      X2 = X/2.0D0
      Y2 = Y/2.0D0
      P1 = X2*X2
      P2 = Y2*Y2
      T1 = -(DLOG(P1+P2)/2.0D0+0.5772156649015329D0)}
   x2:= x/2.0;
   y2:= y/2.0;
   p1:= x2*x2;
   p2:= y2*y2;
   t1:= -(ln(p1+p2)/2.0+0.5772156649015329);
   
{C THE CONSTANT IN THE PRECEDING STATEMENT IS EULER*S
C CONSTANT
      T2 = -DATAN2(Y,X)
      X2 = P1 - P2
      Y2 = X*Y2
      RTERM = 1.0D0
      ITERM = 0.0D0
      RE0 = T1
      IM0 = T2
      T1 = T1 + 0.5D0
      RE1 = T1
      IM1 = T2
      P2 = DSQRT(R2)
      L = 2.106D0*P2 + 4.4D0
      IF (P2.LT.8.0D-1) L = 2.129D0*P2 + 4.0D0}
   if x<>0.0 then 
      t2:= -arctan(y/x)
   else if y>0.0 then
      t2:= -pi/2.0
   else
      t2:= pi/2.0;
   x2:= p1-p2;
   y2:= x*y2;
   rterm:= 1.0;
   iterm:= 0.0;
   re0:= t1;
   im0:= t2;
   t1:= t1 +0.5;
   re1:= t1;
   im1:= t2;
   p2:= sqrt(r2);
   l:= trunc(2.106*p2+4.4);
   if p2<8.0E-1 then l:= trunc(2.129*p2+4.0);
   
{      DO 20 N=1,L
        P1 = N
        P2 = N*N
        R1 = RTERM
        RTERM = (R1*X2-ITERM*Y2)/P2
        ITERM = (R1*Y2+ITERM*X2)/P2
        T1 = T1 + 0.5D0/P1
        RE0 = RE0 + T1*RTERM - T2*ITERM
        IM0 = IM0 + T1*ITERM + T2*RTERM
        P1 = P1 + 1.0D0
        T1 = T1 + 0.5D0/P1
        RE1 = RE1 + (T1*RTERM-T2*ITERM)/P1
        IM1 = IM1 + (T1*ITERM+T2*RTERM)/P1
   20 CONTINUE}
   for n:=1 to l do
      begin
      p1:= n;
      p2:= n*n;
      r1:= rterm;
      rterm:= (r1*x2-iterm*y2)/p2;
      iterm:= (r1*y2+iterm*x2)/p2;
      t1:= t1+0.5/p1;
      re0:= re0 + t1*rterm - t2*iterm;
      im0:= im0 + t1*iterm + t2*rterm;
      p1:= p1 + 1.0;
      t1:= t1 + 0.5/p1;
      re1:= re1 + (t1*rterm-t2*iterm)/p1;
      im1:= im1 + (t1*iterm+t2*rterm)/p1;
      end;
      
{      R1 = X/R2 - 0.5D0*(X*RE1-Y*IM1)
      R2 = -Y/R2 - 0.5D0*(X*IM1+Y*RE1)
      P1 = DEXP(X)
      RE0 = P1*RE0
      IM0 = P1*IM0
      RE1 = P1*R1
      IM1 = P1*R2
      RETURN}
   r1:= x/r2 - 0.5*(x*re1-y*im1);
   r2:= -y/r2 - 0.5*(x*im1+y*re1);
   p1:= exp(x);
   re0:= p1*re0;
   im0:= p1*im0;
   re1:= p1*r1;
   im1:= p1*r2;
   exit;
   end
else if r2<1.96E2 then
   begin   
{C THIS SECTION CALCULATES THE FUNCTIONS USING THE INTEGRAL
C REPRESENTATION, EQN 3, EVALUATED WITH 15 POINT GAUSS-
C HERMITE QUADRATURE
   30 X2 = 2.0D0*X
      Y2 = 2.0D0*Y
      R1 = Y2*Y2
      P1 = DSQRT(X2*X2+R1)
      P2 = DSQRT(P1+X2)
      T1 = EXSQ(1)/(2.0D0*P1)
      RE0 = T1*P2
      IM0 = T1/P2
      RE1 = 0.0D0
      IM1 = 0.0D0}
   x2:= 2.0*X;
   Y2:= 2.0*Y;
   R1:= Y2*Y2;
   P1:= SQRT(X2*X2+R1);
   P2:= SQRT(P1+X2);
   T1:= EXSQ[1]/(2.0*P1);
   RE0:= T1*P2;
   IM0:= T1/P2;
   RE1:= 0.0;
   IM1:= 0.0;
   
{      DO 40 N=2,8
        T2 = X2 + TSQ(N)
        P1 = DSQRT(T2*T2+R1)
        P2 = DSQRT(P1+T2)
        T1 = EXSQ(N)/P1
        RE0 = RE0 + T1*P2
        IM0 = IM0 + T1/P2
        T1 = EXSQ(N)*TSQ(N)
        RE1 = RE1 + T1*P2
        IM1 = IM1 + T1/P2
   40 CONTINUE}
   for n:= 1 to 8 do
      begin
      t2:= x2 + tsq[n];
      p1:= sqrt(t2*t2+r1);
      p2:= sqrt(p1+t2);
      t1:= exsq[n]/p1;
      re0:= re0 + t1*p2;
      im0:= im0 + t1/p2;
      t1:= exsq[n]*tsq[n];
      re1:= re1 + t1*p2;
      im1:= im1 + t1/p2;
      end;
      
{      T2 = -Y2*IM0
      RE1 = RE1/R2
      R2 = Y2*IM1/R2
      RTERM = 1.41421356237309D0*DCOS(Y)
      ITERM = -1.41421356237309D0*DSIN(Y)
C THE CONSTANT IN THE PREVIOUS STATEMENTS IS,OF COURSE,
C SQRT(2.0).
      IM0 = RE0*ITERM + T2*RTERM
      RE0 = RE0*RTERM - T2*ITERM
      T1 = RE1*RTERM - R2*ITERM
      T2 = RE1*ITERM + R2*RTERM
      RE1 = T1*X + T2*Y
      IM1 = -T1*Y + T2*X
      RETURN}
   t2:= -y2*im0;
   re1:= re1/r2;
   r2:= y2*im1/r2;
   rterm:= 1.41421356237309*cos(y);
   iterm:= 1.41421356237309*sin(y);
   im0:= re0*iterm + t2*rterm;
   re0:= re0*rterm - t2*iterm;
   t1:= re1*rterm - r2*iterm;
   t2:= re1*iterm + r2*rterm;
   re1:= t1*x + t2*y;
   im1:= -t1*y + t2*x;
   exit;
   end
else
   begin
   
{C THIS SECTION CALCULATES THE FUNCTIONS USING THE
C ASYMPTOTIC EXPANSIONS
   50 RTERM = 1.0D0
      ITERM = 0.0D0
      RE0 = 1.0D0
      IM0 = 0.0D0
      RE1 = 1.0D0
      IM1 = 0.0D0
      P1 = 8.0D0*R2
      P2 = DSQRT(R2)
      L = 3.91D0+8.12D1/P2
      R1 = 1.0D0
      R2 = 1.0D0
      M = -8
      K = 3}
   rterm:= 1.0;
   iterm:= 0.0;
   re0:= 1.0;
   im0:= 0.0;
   re1:= 1.0;
   im1:= 0.0;
   p1:= 8.0*r2;
   p2:= sqrt(r2);
   l:= trunc(3.91+8.12E1/p2);
   r1:= 1.0;
   r2:= 1.0;
   m:= -8;
   k:= 3;
   
{      DO 60 N=1,L
        M = M + 8
        K = K - M
        R1 = FLOAT(K-4)*R1
        R2 = FLOAT(K)*R2
        T1 = FLOAT(N)*P1
        T2 = RTERM
        RTERM = (T2*X+ITERM*Y)/T1
        ITERM = (-T2*Y+ITERM*X)/T1
        RE0 = RE0 + R1*RTERM
        IM0 = IM0 + R1*ITERM
        RE1 = RE1 + R2*RTERM
        IM1 = IM1 + R2*ITERM
   60 CONTINUE}
   for n:= 1 to l do
      begin
      m:= m + 8;
      k:= k - m;
      r1:= (k-4)*r1;
      r2:= k*r2;
      t1:= n*p1;
      t2:= rterm;
      rterm:= (t2*x+iterm*y)/t1;
      iterm:= (-t2*y+iterm*x)/t1;
      re0:= re0 + r1*rterm;
      im0:= im0 + r1*iterm;
      re1:= re1 + r2*rterm;
      im1:= im1 + r2*iterm;
      end;
      
{      T1 = DSQRT(P2+X)
      T2 = -Y/T1
      P1 = 8.86226925452758D-1/P2
C THIS CONSTANT IS SQRT(PI)/2.0, WITH PI=3.14159...
      RTERM = P1*DCOS(Y)
      ITERM = -P1*DSIN(Y)
      R1 = RE0*RTERM - IM0*ITERM
      R2 = RE0*ITERM + IM0*RTERM
      RE0 = T1*R1 - T2*R2
      IM0 = T1*R2 + T2*R1
      R1 = RE1*RTERM - IM1*ITERM
      R2 = RE1*ITERM + IM1*RTERM
      RE1 = T1*R1 - T2*R2
      IM1 = T1*R2 + T2*R1
      RETURN
99999 FORMAT (42H  ARGUMENT OF THE BESSEL FUNCTIONS IS ZERO,
     * 35H OR LIES IN LEFT HALF COMPLEX PLANE)
      END}
   t1:= sqrt(p2+x);
   t2:= -y/t1;
   p1:= 8.86226925452758E-1/p2;
   rterm:= p1*cos(y);
   iterm:= -p1*sin(y);
   r1:= re0*rterm - im0*iterm;
   r2:= re0*iterm + im0*rterm;
   re0:= t1*r1 - t2*r2;
   im0:= t1*r2 + t2*r1;
   r1:= re1*rterm - im1*iterm;
   r2:= re1*iterm + im1*rterm;
   re1:= t1*r1 - t2*r2;
   im1:= t1*r2 + t2*r1;
   end;
   
end;

end.

