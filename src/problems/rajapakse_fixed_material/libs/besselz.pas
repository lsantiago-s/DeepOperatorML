Unit besselz;

interface
uses complex;

procedure CJYHBS(Z: tcomplex; KODE: integer; 
                 var CJ0,CJ1,CY0,CY1,CH0,CH1: tcomplex);



implementation

procedure CJYHBS(Z: tcomplex; KODE: integer; 
                 var CJ0,CJ1,CY0,CY1,CH0,CH1: tcomplex);
{C
C     WRITTEN BY D.E. AMOS AND S.L. DANIEL
C
C     REFERENCES
C         SLA-73-0262
C
C         NBS HANDBOOK OF MATHEMATICAL FUNCTIONS, AMS 55, BY
C         M. ABRAMOWITZ AND I.A. STEGUN, DECEMBER, 1955, PP. 364, 497.
C
C     ABSTRACT
C         CJYHBS COMPUTES BESSEL FUNCTIONS J/SUB(NU)/(Z), Y/SUB(NU)/(Z),
C         AND STRUVE FUNCTIONS H/SUB(NU)/(Z), FOR COMPLEX Z AND NU=0 OR
C         1. BACKWARD RECURSION IS USED FOR THE J BESSEL FUNCTIONS OF
C         INTEGER ORDER TO SUM THE NEUMANN SERIES FOR THE Y AND H
C         FUNCTIONS FOR 0.LT.modulus(Z).LT.30. FOR modulus(Z).GT.30 THE
C         ASYMPTOTIC EXPANSIONS ARE USED. FOR Z, modulus(Z).GT.0.
C         AND -PI.LT.ARG(Z).LE.PI
C
C     DESCRIPTION OF ARGUMENTS
C
C         INPUT
C           KODE   - A PARAMETER TO SELECT THE PROPER FUNCTION PAIRS
C                    KODE=1 RETURNS J0,J1,Y0,Y1 FUNCTIONS
C                    KODE=2 RETURNS J0,J1,H0,H1 FUNCTIONS
C                    KODE=3 RETURNS J0,J1,Y0,Y1,H0,H1 FUNCTIONS
C           Z      - COMPLEX ARGUMENT, Z.NE.CMPLX(0.,0.)
C                    AND -PI.LT.ARG(Z).LE.PI
C
C         OUTPUT
C           CJ0    - BESSEL FUNCTION J/SUB(0)/(Z), A COMPLEX NUMBER
C           CJ1    - BESSEL FUNCTION J/SUB(1)/(Z), A COMPLEX NUMBER
C           CY0    - BESSEL FUNCTION Y/SUB(0)/(Z), A COMPLEX NUMBER
C           CY1    - BESSEL FUNCTION Y/SUB(1)/(Z), A COMPLEX NUMBER
C           CH0    - STRUVE FUNCTION H/SUB(0)/(Z), A COMPLEX NUMBER
C           CH1    - STRUVE FUNCTION H/SUB(1)/(Z), A COMPLEX NUMBER
C
C     ERROR CONDITIONS
C         ERROR #1, Z=0  ON INPUT, A FATAL ERROR
C         ERROR #2, KODE NOT 1 OR 2 OR 3 ON INPUT, A FATAL ERROR
C
C
C
C     OTHER ROUTINES CALLED: (NONE)
C}
var
{      COMPLEX Z,CJ0,CJ1,CY0,CY1,ZLAM(70),CAPJ(71),CAPR(71),SN,CONE,CTWO,
     1TN,C1,C2,C3,C4,C5,C6,CR,TOPI,FOPI,CSUM,CK,SK,CZERO,CH0,CH1,Z1
      DATA EULER/5.77215664901533E-1/
      DATA PIO2/1.57079632679490/
      DATA ISET/1/}
      
zlam: array [1..70] of tcomplex;
capj, capr: array [1..71] of tcomplex;
sn, cone, ctwo, tn, c1, c2, c3, c4, c5, c6, cr, topi, fopi, csum, ck, 
sk, z1: tcomplex;
zabs, trpi, rtopi, x, y, theta, theta1, sgn, fn, ay, ey, ax: double;
i, j, k, l, nu, nup1,k0, k1, kh: integer;

const
pio2= pi/2.0;
EULER= 5.77215664901533E-1;

{C
      ZABS=modulus(Z)
      IF(ZABS.EQ.0.) GO TO 90
      IF(KODE.LT.1.OR.KODE.GT.3) GO TO 91}
begin
zabs:= modulus(z);
if zabs=0.0 then
   begin
   writeln('CJYHBS: Error: z equals 0');
   exit;
   end;
if (kode<1) or (kode>3) then
   begin
   writeln('CJYHBS: Error: invalid KODE ',KODE);
   exit;
   end;
   
{      GO TO (1,2),ISET
    1 ISET=2
C
      C1=CMPLX(0.,-2.)
      C2=CMPLX(-2.,0.)
      C3=CMPLX(0.,2.)
      C4=CMPLX(2.,0.)
      K=1
      DO 3 J=1,17
      ZLAM(K)=C1
      ZLAM(K+1)=C2
      ZLAM(K+2)=C3
      ZLAM(K+3)=C4
    3 K=K+4
      ZLAM(K)=C1
      ZLAM(K+1)=C2
      PI=3.14159265358979
      TRPI=2./PI
      TOPI=CMPLX(TRPI,0.)
      FOPI=CMPLX(4./PI,0.)
      RTOPI=SQRT(TRPI)
      CTWO=CMPLX(2.,0.)
      CONE=CMPLX(1.,0.)
      CZERO=CMPLX(0.,0.)
    2 CONTINUE}
C1:=CMPLX(0.0,-2.0);
C2:=CMPLX(-2.0,0.0);
C3:=CMPLX(0.0,2.0);
C4:=CMPLX(2.0,0.0);
k:= 1;
for j:= 1 to 17 do
   begin
   zlam[k]:= c1;
   zlam[k+1]:= c2;
   zlam[k+2]:= c3;
   zlam[k+3]:= c4;
   k:= k+4;
   end;
zlam[k]:= c1;
zlam[k+1]:= c2;
trpi:= 2.0/pi;
topi:= cmplx(trpi,0.0);
fopi:= cmplx(4.0/pi, 0.0);
rtopi:= sqrt(trpi);
ctwo:= cmplx(2.0,0.0);
cone:= cmplx(1.0,0.0);

    
{      X=REAL(Z)
      Y=AIMAG(Z)
      IF(X) 4,5,6
    4 IF(Y) 9,8,7
    9 THETA1=ATAN(Y/X)
      THETA=THETA1-PI
      SGN=-2.
      GO TO 50
    5 IF(Y) 11,90,13
   11 THETA=-PIO2
      THETA1=THETA
      GO TO 50
   13 THETA=PIO2
      THETA1=THETA
      GO TO 50
    7 THETA1=ATAN(Y/X)
      THETA=THETA1+PI
      SGN=2.
      GO TO 50
 8    THETA1=0.
      THETA=PI
      SGN=2.
      GO TO 50
    6 THETA1=ATAN(Y/X)
      THETA=THETA1
   50 CONTINUE}
x:= re(z);
y:= im(z);
if x=0.0 then
   if y<0.0 then
      begin
      theta:= -pio2;
      theta1:= theta;
      end
   else
      begin
      theta:= pio2;
      theta1:= theta;
      end
else
   begin
   if y=0.0 then theta1:= 0.0
   else theta1:= arctan(y/x);
   if (x<0.0) and (y>0.0) then
      begin
      theta:= theta1+pi;
      sgn:= 2.0;
      end;
   if (x<0.0) and (y=0.0) then
      begin
      theta:= pi;
      sgn:= 2.0;
      end;
   if (x<0.0) and (y<0.0) then
      begin
      theta:= theta1-pi;
      sgn:= -2.0;
      end;
   if x>0.0 then
      theta:= theta1;
   end;

{C
C     BACKWARD RECURSION FOR J/SUBK/(X),K=0,1,...
C
      IF(ZABS.GE.30.) GO TO 200
      NU=IFIX(ZABS)+40
      NUP1=NU+1
      FN=NU
      AY=ABS(Y)
      Z1=CMPLX(X,AY)
      SN=CZERO
      CAPR(NUP1)=SN
      TN=CMPLX(FN+FN,0.)
      DO 10 I=1,NU
      L=NUP1-I
      CR=Z1/(TN-Z1*CAPR(L+1))
      CAPR(L)=CR
      SN=CR*(ZLAM(L)+SN)
      TN=TN-CTWO
   10 CONTINUE
      SN=SN+CONE
      EY=EXP(AY)
      CAPJ(1)=CMPLX(EY*COS(X),-EY*SIN(X))/SN
      DO 15 I=1,NU
      CAPJ(I+1)=CAPJ(I)*CAPR(I)
   15 CONTINUE
      IF(Y.GE.0.) GO TO 17
      DO 16 I=1,NUP1
   16 CAPJ(I)=CONJG(CAPJ(I))
 17   CJ1=CAPJ(2)
      CJ0=CAPJ(1)
      IF(KODE.EQ.2) GO TO 100}
if zabs<30.0 then
   begin
   nu:= trunc(zabs)+40;
   nup1:= nu+1;
   fn:= nu;
   ay:= abs(y);
   z1:= cmplx(x,ay);
   sn:= czero;
   capr[nup1]:= sn;
   tn:= cmplx(fn+fn, 0.0);
   for i:= 1 to nu do
      begin
      l:= nup1-i;
      cr:= z1/(tn-z1*capr[l+1]);
      capr[l]:= cr;
      sn:= cr*(zlam[l]+sn);
      tn:= tn-ctwo;
      end;
   sn:= sn+cone;
   ey:= exp(ay);
   capj[1]:= cmplx(ey*cos(x), -ey*sin(x))/sn;
   for i:= 1 to nu do
      capj[i+1]:= capj[i]*capr[i];
   if y<0.0 then 
      for i:=1 to nup1 do
         capj[i].y:= -capj[i].y;
   cj1:= capj[2];
   cj0:= capj[1];
   if kode<>2 then
      begin     

{C
C     NEUMANN SERIES FOR Y0
C
      K0=(NU-1)/2
      K1=(NU-2)/2
      C1=CMPLX(ALOG(ZABS*.5)+EULER,THETA)
      C2=C1-CONE
      SK=-CONE
      CK=CONE
      CSUM=CZERO
      DO 20 I=1,K0
      CSUM=CSUM+SK*CAPJ(I+I+1)/CK
      SK=-SK
      CK=CK+CONE
   20 CONTINUE
      CY0=TOPI*(C1*CAPJ(1)-(CSUM+CSUM))}
      k0:= (nu-1) div 2;
      k1:= (nu-2) div 2;
      c1:= cmplx(ln(zabs*0.5)+euler, theta);
      c2:= c1-cone;
      sk:= -cone;
      ck:= cone;
      csum:= czero;
      for i:=1 to k0 do
         begin
	 csum:= csum+sk*capj[i+i+1]/ck;
	 sk:= -sk;
	 ck:= ck+cone;
	 end;
      cy0:= topi*(c1*capj[1]-(csum+csum));
      
{C
C     NEUMANN SERIES FOR Y1
C
      C3=CTWO
      C4=CTWO+CONE
      C5=CONE
      SK=-CONE
      CSUM=CZERO
      DO 25 I=1,K1
      CSUM=CSUM+SK*(C4/C3)*CAPJ(I+I+2)/C5
      SK=-SK
      C3=C3+CONE
      C4=C4+CTWO
      C5=C5+CONE
   25 CONTINUE
      CY1=TOPI*(-CAPJ(1)/Z+C2*CAPJ(2)-CSUM)
      IF(KODE.EQ.1) RETURN}
      c3:= ctwo;
      c4:= ctwo+cone;
      c5:= cone;
      sk:= -cone;
      csum:= czero;
      for i:= 1 to k1 do
         begin
	 csum:= csum+sk*(c4/c3)*capj[i+i+2]/c5;
	 sk:= -sk;
	 c3:= c3+ cone;
	 c4:= c4+ ctwo;
	 c5:= c5+ cone;
	 end;
      cy1:= topi*(-capj[1]/z+c2*capj[2]-csum);
      if kode=1 then
         exit;
      end;
      
{C
C     NEUMANN SERIES FOR H0 AND H1
C
  100 CONTINUE
      KH=NU/2
      CK=CONE+CTWO
      C1=CONE
      C4=CZERO
      C3=C4
      L=2
      DO 30 K=1,KH
      C3=C3+CAPJ(L)/C1
      C4=C4+CAPJ(L+1)/(C1*CK)
      C1=CK
      CK=CK+CTWO
      L=L+2
   30 CONTINUE
      CH0=FOPI*C3
      CH1=TOPI*(CONE-CAPJ(1)+C4+C4)
      RETURN}
   kh:= nu div 2;
   ck:= cone+ctwo;
   c1:= cone;
   c4:= czero;
   c3:= c4;
   l:=2;
   for k:=1 to kh do
      begin
      c3:= c3+capj[l]/c1;
      c4:= c4+capj[l+1]/(c1*ck);
      c1:= ck;
      ck:= ck+ctwo;
      l:= l+2;
      end;
   ch0:= fopi*c3;
   ch1:= topi*(cone-capj[1]+c4+c4);
   exit;
   end;

{C
  200 CONTINUE
C
C     ASYMPTOTIC EXPANSIONS FOR Y0,Y1,J0,J1,ABS(Z).GE.30
C
      THETA1=THETA1*.5
      CR=CMPLX(COS(THETA1),-SIN(THETA1))
      CR=(RTOPI/SQRT(ZABS))*CR
C     FOR ABS(Z).GT.30
      IF(X.GE.0.) GO TO 207
      Y=-Y
 207  CONTINUE
      AX=ABS(X)
      Z1=CMPLX(AX,Y)}
theta1:= theta1*0.5;
cr:= cmplx(cos(theta1),-sin(theta1));
cr:= (rtopi/sqrt(zabs))*cr;
if x<0.0 then y:= -y;
ax:= abs(x);
z1:= cmplx(ax, y);      
      
{      DO 205 L=1,2
      NU=L-1
      TN=4*NU*NU
      SN=CMPLX(AX-(FLOAT(NU)*.5+.25)*PI,Y)
      C2=CZERO
      CK=CONE
      C1=CONE
      SK=CONE
      C4=8.*Z1
      C3=C4
      DO 40 K=1,16
      CK=CK*((TN-SK*SK)/C3)
      IF(MOD(K,2).EQ.0) GO TO 42
      C2=C2+CK
      CK=-CK
      GO TO 44
   42 C1=C1+CK
      GO TO 44
   44 IF(modulus(CK).LT.1.E-13) GO TO 45
      C3=C3+C4
      SK=SK+CTWO
   40 CONTINUE
   45 CONTINUE
      C5=CCOS(SN)
      C6=CSIN(SN)
      GO TO (201,202),L
 201  CY0=CR*(C1*C6+C2*C5)
      CJ0=CR*(C1*C5-C2*C6)
      GO TO 205
 202  CY1=CR*(C1*C6+C2*C5)
      CJ1=CR*(C1*C5-C2*C6)
  205 CONTINUE
      IF(X.GE.0.) GO TO 206}
for l:=1 to 2 do
   begin
   nu:= l-1;
   tn:= 4*nu*nu;
   sn:= cmplx(ax- (0.5*nu +0.25)*pi, y);
   c2:= czero;
   ck:= cone;
   c1:= cone;
   sk:= cone;
   c4:= 8.0*z1;
   c3:= c4;
   for k:=1 to 16 do
      begin
      ck:= ck* ((tn-sk*sk)/c3);
      if k mod 2 <> 0 then
         begin
         c2:= c2+ck;
         ck:= -ck;
         end
      else
         c1:= c1+ck;
      
      if modulus(ck)<1.0E-13 then break;
      c3:= c3+c4;
      sk:= sk+ctwo;
      end;
   c5:= ccos(sn);
   c6:= csin(sn);
   if l=1 then
      begin
      cy0:= cr*(c1*c6+c2*c5);
      cj0:= cr*(c1*c5-c2*c6);
      end
   else
      begin
      cy1:= cr*(c1*c6+c2*c5);
      cj1:= cr*(c1*c5-c2*c6);
      end;
   end;
if x<0.0 then
   begin

{C     FORMULA FOR Z=-Z1, -PI/2..LT.ARG(Z1).LT.PI/2.
      CY0=CY0+CMPLX(0.0,SGN)*CJ0
      CY1=-CY1-CMPLX(0.0,SGN)*CJ1
      CJ1=-CJ1
  206 CONTINUE}
   cy0:= cy0+cmplx(0.0, sgn)*cj0;
   cy1:= -cy1-cmplx(0.0, sgn)*cj1;
   cj1:= -cj1;
   end;
{      IF(KODE.EQ.1) RETURN}
if kode=1 then exit;

{C
C     ASYMPTOTIC EXPANSIONS FOR H0,H1,ABS(Z).GE.30
C
      CK=CONE/Z
      CSUM=CK
      SK=CONE
      CR=CK*CK
      DO 60 K=1,8
      CK=-CK*SK*SK*CR
      CSUM=CSUM+CK
      IF(modulus(CK).LT.1.E-14) GO TO 62
      SK=SK+CTWO
   60 CONTINUE}
ck:= cone/z;
csum:= ck;
sk:= cone;
cr:= ck*ck;
for k:=1 to 8 do
   begin
   ck:= -ck*sk*sk*cr;
   csum:= csum+ck;
   if modulus(ck)<1.0E-14 then break;
   sk:= sk+ctwo;
   end; 
   
{ 62   CH0=CY0+TOPI*CSUM
      CK=CR
      CSUM=CR
      C1=CONE
      C2=C1+CTWO
      DO 64 K=1,8
      CK=-CK*(C1*C2*CR)
      CSUM=CSUM+CK
      IF(modulus(CK).LT.1.E-14) GO TO 66
      C1=C2
      C2=C2+CTWO
   64 CONTINUE}
ch0:= cy0+topi*csum;
ck:= cr;
csum:= cr;
c1:= cone;
c2:= c1+ ctwo;
for k:=1 to 8 do
   begin
   ck:= -ck*(c1*c2*cr);
   csum:= csum+ck;
   if modulus(ck)<1.0E-14 then break;
   c1:= c2;
   c2:= c2+ctwo;
   end;

{ 66   CH1=CY1+TOPI*(CONE+CSUM)
      RETURN
 90   CALL XERROR('CJHYBS, Z IS ZERO',17,1,2)
      RETURN
 91   CALL XERROR('CJHYBS, KODE NOT 1, 2, OR 3',27,2,2)
      RETURN
      END}
      
ch1:= cy1+topi*(cone+csum);
end;

end.

