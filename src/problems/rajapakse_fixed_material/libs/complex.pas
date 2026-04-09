{*****************************************************************************}
{ File                   : complex.pas
  Author                 : Mazen NEIFER
  Creation date          : 21/09/2000
  Last modification date : 27/09/2000
  Licence                : GPL
  Bug report             : mazen_nefer@ayna.com                               }
{*****************************************************************************}
UNIT Complex;
INTERFACE
TYPE
  TReal=double;{This can be changed to any real type to support more huge values}
  PReal=^TReal;
  PComplex=^TComplex;
  TComplex=RECORD
    x,y:Real;{It's too simple a complex is a couple of reals!}
  END;
  PMatrix=^TMatrix;
  TMatrix=RECORD
    n,p:Byte;{It will be nice if we can use "TMatrix=ARRAY[n,p]OF TComplex;"}
    Values:PComplex;{Till this will be possible we can usally use this}
  END;
CONST
  c_i:TComplex=(x:0;y:1);{And what about the solution of x^2+1=0!!!!!?}
  czero: TComplex=(x:0;y:0);
  Digit:Byte=3;{I prefer 3 zeros but you can change it}
OPERATOR :=(r:Real)RESULT:TComplex;
OPERATOR -(z:TComplex)RESULT:TComplex;
OPERATOR +(z1,z2:TComplex)RESULT:TComplex;
OPERATOR -(z1,z2:TComplex)RESULT:TComplex;
OPERATOR *(z1,z2:TComplex)RESULT:TComplex;
OPERATOR /(z1,z2:TComplex)RESULT:TComplex;
OPERATOR * (z:TComplex;n:Byte)RESULT:TComplex;
OPERATOR **(z:TComplex;n:Byte)RESULT:TComplex;
OPERATOR /(z:TComplex;n:Byte)RESULT:TComplex;
OPERATOR /(z:TComplex;n:DWord)RESULT:TComplex;
OPERATOR +(r:Real;z:TComplex)RESULT:TComplex;
OPERATOR *(r:Real;z:TComplex)RESULT:TComplex;
OPERATOR /(z:TComplex;r:Real)RESULT:TComplex;
OPERATOR =(z:TComplex;r:Real)RESULT:Boolean;
FUNCTION cexp(z1:TComplex):TComplex;
FUNCTION Modulus(z:TComplex):Real;
FUNCTION argument(z: tcomplex): double;
FUNCTION log(z1: tcomplex): tcomplex;
function csqrt(z1: tcomplex):tcomplex;
function csqr(z1: tcomplex): tcomplex;
PROCEDURE ReadComplex(VAR z:TComplex);
PROCEDURE WriteComplex(z:TComplex);
function cmplx(x,y:double):tcomplex;
function re(z:tcomplex): double;
function im(z:tcomplex): double;
function csin(z1: tcomplex): tcomplex;
function ccos(z1: tcomplex): tcomplex;
{***************************Matrix********************************}
FUNCTION Matrix(_n,_p:Byte):TMatrix;
PROCEDURE KillMatrix(VAR aMatrix:TMatrix);
OPERATOR +(M1,M2:TMatrix)RESULT:TMatrix;
OPERATOR -(M1,M2:TMatrix)RESULT:TMatrix;
OPERATOR *(M1,M2:TMatrix)RESULT:TMatrix;
OPERATOR /(M1,M2:TMatrix)RESULT:TMatrix;
FUNCTION Det(aMatrix:TMatrix):TComplex;
FUNCTION Cofactor(aMatrix:TMatrix;k,l:Byte):TMatrix;
PROCEDURE ReadMatrix(n,p:Byte;VAR M:TMatrix);
PROCEDURE WriteMatrix(M:TMatrix);
IMPLEMENTATION
OPERATOR :=(r:Real)RESULT:TComplex;
  BEGIN
    WITH RESULT DO
      BEGIN
        x:=r;
        y:=0;
      END;
  END;
OPERATOR -(z:TComplex)RESULT:TComplex;
  BEGIN
    WITH RESULT DO
      BEGIN
        x:=-z.x;
        y:=-z.y;
      END;
  END;
OPERATOR +(z1,z2:TComplex)RESULT:TComplex;
  BEGIN
    WITH RESULT DO
      BEGIN
        x:=z1.x+z2.x;
        y:=z1.y+z2.y;
      END;
  END;
OPERATOR -(z1,z2:TComplex)RESULT:TComplex;
  BEGIN
    WITH RESULT DO
      BEGIN
        x:=z1.x-z2.x;
        y:=z1.y-z2.y;
      END;
  END;
OPERATOR *(z1,z2:TComplex)RESULT:TComplex;
   BEGIN
     WITH RESULT DO
       BEGIN
         x:=z1.x*z2.x-z1.y*z2.y;
         y:=z1.x*z2.y+z1.y*z2.x;
       END;
   END;
OPERATOR /(z1,z2:TComplex)RESULT:TComplex;
{  VAR
    M:Real;
  BEGIN
    M:=modulus(z2);
    WITH RESULT DO
      BEGIN
        x:=(z1.x*z2.x+z1.y*z2.y)/M;
        y:=(z1.y*z2.x-z1.x*z2.y)/M;
      END;
  END;}
var
t, den: double;
begin
if abs(z2.x)>=abs(z2.y) then
   begin
   t:= z2.y/z2.x;
   den:= z2.x + t*z2.y;
   result.x:= (z1.x+z1.y*t)/den;
   result.y:= (z1.y-z1.x*t)/den;
   end
else
   begin
   t:= z2.x/z2.y;
   den:= z2.y + t*z2.x;
   result.x:= (z1.x*t+z1.y)/den;
   result.y:= (z1.y*t-z1.x)/den;
   end

end;

OPERATOR *(z:TComplex;n:Byte)RESULT:TComplex;
  BEGIN
    WITH RESULT DO
      BEGIN
        x:=z.x*n;
        y:=z.y*n;
      END;
  END;
OPERATOR **(z:TComplex;n:Byte) RESULT:TComplex;
  VAR
    i:Byte;
  BEGIN
    WITH RESULT DO
      BEGIN
        x:=1;
        y:=0;
        FOR i:=1 TO n DO
          BEGIN
            x:=x*z.x-y*z.y;
            y:=x*z.y+y*z.x;
          END;
      END;
  END;
OPERATOR /(z:TComplex;n:Byte)RESULT:TComplex;
  BEGIN
    WITH RESULT DO
      BEGIN
        x:=z.x/n;
        y:=z.y/n;
      END;
  END;
OPERATOR /(z:TComplex;n:DWord)RESULT:TComplex;
  BEGIN
    WITH RESULT DO
      BEGIN
        x:=z.x/n;
        y:=z.y/n;
      END;
  END;
OPERATOR +(r:Real;z:TComplex)RESULT:TComplex;
  BEGIN
    WITH RESULT DO
      BEGIN
        x:=z.x+r;
        y:=z.y;
      END;
  END;
OPERATOR *(r:Real;z:TComplex)RESULT:TComplex;
  BEGIN
    WITH RESULT DO
      BEGIN
        x:=r*z.x;
        y:=r*z.y;
      END;
  END;
OPERATOR /(z:TComplex;r:Real)RESULT:TComplex;
  BEGIN
    WITH RESULT DO
      BEGIN
        x:=z.x/r;
        y:=z.y/r;
      END;
  END;
OPERATOR =(z:TComplex;r:Real)RESULT:Boolean;
  BEGIN
    WITH z DO
      RESULT:=(x=r)AND(y=0);
  END;
FUNCTION cexp(z1:TComplex):TComplex;
{  CONST
    MaxLevel=20;
  VAR
    k:Byte;
    RESULT:TComplex;
  BEGIN
    RESULT:=1;
    FOR k:=MaxLevel DOWNTO 1 DO
        RESULT:=1.0 + RESULT*(z/k);
    exp:=RESULT;
  END;}
var
t: double;
result: tcomplex;
begin
{writeln(z1.x);}
if z1.x<-750.0 then t:=0.0 else t:=exp(z1.x);
if z1.y=0.0 then with result do
   begin
   x:=t;
   y:=0.0;
   cexp:= result;
   exit;
   end;
with result do
   begin
   x:= t* cos(z1.y);
   y:= t* sin(z1.y);
   end;
cexp:= result;
end;

FUNCTION Modulus(z:TComplex):Real;
{  BEGIN
    WITH z DO
      modulus:=sqrt(sqr(x)+sqr(y));
  END;}
begin
with z do
   if (x=0.0) or (y=0.0) then
      modulus:= abs(x + y)
   else
      modulus:= sqrt( sqr(x) + sqr(y));
end;

function argument(z: tcomplex): double;
var
t: double;

begin
with z do
   if x = 0.0 then
      if y>=0.0 then argument:= pi/2.0
      else argument:= -pi/2.0
   else if y=0.0 then
      if x>=0.0 then argument:=0.0
      else argument:= pi
   else
      begin
      t:= arctan(y/x);
      if t<0.0 then
         if x<0.0 then t:=t+pi else t:= t+ 2.0*pi
      else
         if x<0.0 then t:=t+pi;
      argument:= t;
      end;
end;


FUNCTION log(z1: tcomplex): tcomplex;
var
result: tcomplex;
begin
if z1.y=0.0 then with result do
   begin
   x:=ln(z1.x);
   y:=0.0;
   log:= result;
   exit;
   end;
with result do
   begin
   x:= ln(modulus(z1));
   y:= argument(z1);
   end;
log:= result;
end;

function csqrt(z1: tcomplex):tcomplex;
var
r,t: double;
result: tcomplex;
begin
if z1.y=0.0 then with result do
   begin
   t:=sqrt(abs(z1.x));
   if z1.x>=0.0 then
      begin
      x:=t;
      y:=0.0;
      end
    else
      begin
      x:=0.0;
      y:=t;
      end;
   csqrt:= result;
   exit;
   end;
if z1.x=0.0 then with result do
   begin
   r:= sqrt(abs(z1.y));
   x:= r/sqrt(2.0);
   if z1.y>=0.0 then
      y:= x
   else
      y:= -x;
   csqrt:= result;
   exit;
   end;

r:=sqrt(modulus(z1));
t:=z1.y/z1.x;
if z1.x>=0.0 then
   t:=1.0/sqrt(1.0+sqr(t))
else
   t:=-1.0/sqrt(1.0+sqr(t));
with result do
   begin
   x:= r*sqrt(0.5*(1.0+t));
   y:= r*sqrt(0.5*(1.0-t));
   if z1.y<0.0 then
      y:=-y;
   end;
csqrt:= result;
end;

function csqr(z1: tcomplex): tcomplex;
var
result: tcomplex;
begin
with result do
   begin
   x:= sqr(z1.x)-sqr(z1.y);
   y:= 2.0*z1.x*z1.y;
   end;
csqr:= result;
end;

PROCEDURE ReadComplex(VAR z:TComplex);
  BEGIN
    WITH z DO
      Read(x,y);
  END;
{  VAR
    s:STRING;
  BEGIN
    Read(s);
  END;}
PROCEDURE WriteComplex(z:TComplex);
  BEGIN
    WITH z DO
      Write(x:1:Digit,'+i*',y:1:Digit);
  END;

function cmplx(x,y:double):tcomplex;
var
result: tcomplex;
begin
result.x:= x;
result.y:= y;
cmplx:= result;
end;

function re(z:tcomplex): double;
begin
re:= z.x;
end;

function im(z:tcomplex): double;
begin
im:= z.y;
end;

function csin(z1: tcomplex):tcomplex;
var
result: tcomplex;
t: double;
begin
if z1.x=0.0 then with result do
   begin
   x:=sin(z1.x);
   y:=0.0;
   csin:= result;
   exit;
   end;
t:= exp(z1.y);
with result do
   begin
   x:= 0.5*(t+1.0/t)*sin(z1.x);
   y:=-0.5*(1.0/t-t)*cos(z1.x);
   end;
csin:= result;
end;

function ccos(z1: tcomplex): tcomplex;
var
result: tcomplex;
t: double;
begin
if z1.y=0.0 then with result do
   begin
   x:=cos(z1.x);
   y:=0.0;
   ccos:= result;
   exit;
   end;
t:= exp(z1.y);
with result do
   begin
   x:= 0.5*(t+1.0/t)*cos(z1.x);
   y:= 0.5*(1.0/t-t)*sin(z1.x);
   end;
ccos:= result;
end;


FUNCTION Matrix(_n,_p:Byte):TMatrix;
  BEGIN
    WITH Matrix DO
      BEGIN
        n:=_n;
        p:=_p;
        GetMem(Values,n*p*SizeOf(TComplex))
      END;
  END;
PROCEDURE KillMatrix(VAR aMatrix:TMatrix); 
  BEGIN
    WITH aMatrix DO
      BEGIN
        FreeMem(Values,n*p*SizeOf(TComplex));
        n:=0;
        p:=0;
      END;
  END;   
FUNCTION max(n,p:Byte):Byte;
  BEGIN
    IF n<p
    THEN
      max:=p
    ELSE
      max:=n;
  END;
OPERATOR +(M1,M2:TMatrix)RESULT:TMatrix;
  VAR
    i,j:Byte;
    _M,_M1,_M2:PComplex;
  BEGIN
    WITH RESULT DO
      BEGIN
        n:=max(M1.n,m2.n);
        p:=max(M1.p,M2.p);
        RESULT:=Matrix(n,p);
        _M:=Values;
        _M1:=M1.Values;
        _M2:=M2.Values;
        FOR i:=1 TO n DO
          FOR j:=1 TO p DO
            BEGIN
              _M^:=0;
              IF(i<=M1.n)AND(j<=M1.p)
              THEN
                _M^:=_M^+_M1^;
              IF(i<=M2.n)AND(j<=M2.p)
              THEN
                _M^:=_M^+_M2^;
              inc(_M);
              inc(_M1);
              inc(_M2);
            END;
      END;
  END;
OPERATOR -(M1,M2:TMatrix)RESULT:TMatrix;
  VAR
    i,j:Byte;
    _M,_M1,_M2:PComplex;
  BEGIN
    WITH RESULT DO
      BEGIN
        n:=max(M1.n,m2.n);
        p:=max(M1.p,M2.p);
        RESULT:=Matrix(n,p);
        _M:=Values;
        _M1:=M1.Values;
        _M2:=M2.Values;
        FOR i:=1 TO n DO
          FOR j:=1 TO p DO
            BEGIN
              _M^:=0;
              IF(i<=M1.n)AND(j<=M1.p)
              THEN
                _M^:=_M^+_M1^;
              IF(i<=M2.n)AND(j<=M2.p)
              THEN
                _M^:=_M^-_M2^;
              inc(_M);
              inc(_M1);
              inc(_M2);
            END;
      END;
  END;
OPERATOR *(M1,M2:TMatrix)RESULT:TMatrix;
  VAR
    i,j:Byte;
    _M,_M1,_M2:PComplex;
  BEGIN
    WITH RESULT DO
      BEGIN
        n:=max(M1.n,m2.n);
        p:=max(M1.p,M2.p);
        RESULT:=Matrix(n,p);
        _M:=Values;
        _M1:=M1.Values;
        _M2:=M2.Values;
        FOR i:=1 TO n DO
          FOR j:=1 TO p DO
            BEGIN
              _M^:=0;
              IF(i<=M1.n)AND(j<=M1.p)
              THEN
                _M^:=_M^+_M1^;
              IF(i<=M2.n)AND(j<=M2.p)
              THEN
                _M^:=_M^+_M2^;
              inc(_M);
              inc(_M1);
              inc(_M2);
            END;
      END;
  END;
OPERATOR /(M1,M2:TMatrix)RESULT:TMatrix;
  VAR
    i,j:Byte;
    _M,_M1,_M2:PComplex;
  BEGIN
    IF Det(M2)=0
    THEN
      RunError(0);
    WITH RESULT DO
      BEGIN
        n:=max(M1.n,m2.n);
        p:=max(M1.p,M2.p);
        RESULT:=Matrix(n,p);
        _M:=Values;
        _M1:=M1.Values;
        _M2:=M2.Values;
        FOR i:=1 TO n DO
          FOR j:=1 TO p DO
            BEGIN
              _M^:=0;
              IF(i<=M1.n)AND(j<=M1.p)
              THEN
                _M^:=_M^+_M1^;
              IF(i<=M2.n)AND(j<=M2.p)
              THEN
                _M^:=_M^+_M2^;
              inc(_M);
              inc(_M1);
              inc(_M2);
            END;
      END;
  END;
FUNCTION Cofactor(aMatrix:TMatrix;k,l:Byte):TMatrix;
  VAR
    i,j,i2,j2:Byte;
  BEGIN
    WITH aMatrix DO
      Cofactor:=Matrix(n-1,p-1);
    i:=0;
    j:=0;
    i2:=0;
    j2:=0;
    WITH Cofactor DO
      WHILE i<n DO
        BEGIN
	  IF i<>k
	  THEN
	    BEGIN
	      WHILE j<p DO
	        BEGIN
	          IF j<>l
	          THEN
	            BEGIN
	              Values[i2*(n-1)+j2]:=aMatrix.Values[i*n+j];
		      inc(j2);
		    END;
                  inc(j);
                END;
	      inc(i2);;
	    END;
	  inc(i);
        END;	       
  END;
FUNCTION Det(aMatrix:TMatrix):TComplex;
  VAR 
    j:Byte;
  BEGIN
    Det:=0;
    WITH aMatrix DO
      FOR j:=1 TO n DO
        Det:=Det+Det(Cofactor(aMatrix,1,j))*Values[j];
  END;
PROCEDURE WriteMatrix(M:TMatrix);
  VAR
    i,j:Byte;
  BEGIN
    WITH M DO
      FOR i:=0 TO n-1 DO
        BEGIN
	  FOR j:=0 TO p-1 DO
	    BEGIN
	      WriteComplex(Values[i*n+j]);
	      Write(' ');
	    END;
	  WriteLn;
	END;
  END;
PROCEDURE ReadMatrix(n,p:Byte;VAR M:TMatrix);
  VAR
    i,j:Byte;
  BEGIN
    M:=Matrix(n,p);
    WITH M DO
      FOR i:=1 TO n DO
        FOR j:=1 TO n DO
	  BEGIN
	    Write('M[',i,',',j,']=');
	    ReadComplex(Values[i*n+j]);
	  END;
  END;
END .
