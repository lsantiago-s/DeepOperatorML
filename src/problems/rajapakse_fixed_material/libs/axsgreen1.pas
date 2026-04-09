unit axsgreen1;
interface
uses complex, quadpack1, axskern1, besselj, besselk1;

type
mat_const= record
   c11, c12, c33, c13, c44, rho, damp: double;
   end;

function axsgreenfunc(mat: mat_const;
         r, z, h, loadr, loadh, omega: double;
         bvptype, loadtype, component, part: integer): double;

function axsgreenfunc(mat: mat_const;
         r, z, h, loadr, loadh, omega: double;
         bvptype, loadtype, component:integer): tcomplex;

function axsgreensrzz0(mat: mat_const;
         r, z, h, loadr, loadh, omega: double;
         bvptype, loadtype, component, part: integer): double;

function rayleigh_pole(alpha, beta, kappa: double): double;

implementation
const
delta_load_precision: double = 1.0E-03;
const_load_precision: double = 1.0E-04;
smalldouble= 1.0E-10;

function rayleigh_pole(alpha, beta, kappa: double): double;
const
maxit= 1000;
precis= 1.0E-6;
var
r, dr, a, e1, e2: tcomplex;
z, gamma: double;
n: integer;

begin
gamma:= 1.0+alpha*beta - kappa*kappa;
if gamma<=0.0 then
   begin
   rayleigh_pole:=2.0;
   exit;
   end;

z:= 2.0;
a.x:= sqr(gamma*z*z-1.0-alpha)-4.0*alpha*(beta*sqr(z*z)-beta*z*z-z*z+1.0);
a.y:= 0.0;
a:=csqrt(a);

e1.x:= gamma*z*z-1.0-alpha+a.x;
e1.y:= a.y;
e1:=csqrt(e1);
e2.x:= gamma*z*z-1.0-alpha-a.x;
e2.y:= -a.y;
e2:=csqrt(e2);
e1:= e1*e2;
r.x:= (2.0*(1-kappa)*z*z-gamma*z*z+alpha)*(1-z*z)- 0.5*e1.x;
r.y:= -0.5*e1.y;
n:=0;
while (modulus(r)> precis) and (n<=maxit) do
   begin
   n:=n+1;
   dr.x:= 2.0*z*(2.0*z*z*(-2.0+2.0*kappa+gamma)+2.0-2.0*kappa-gamma-alpha);
   dr.y:= 0.0;
   e2.x:= 2.0*z*alpha*(-beta-1.0+2.0*beta*z*z);
   e2.y:= 0.0;
   e2:= e2/e1;
   dr:= dr-e2;

   dr:=r/dr;
   z:= z- dr.x;
   a.x:= sqr(gamma*z*z-1.0-alpha)-4.0*alpha*(beta*sqr(z*z)-beta*z*z-z*z+1.0);
   a.y:= 0.0;
   a:= csqrt(a);

   e1.x:= gamma*z*z-1.0-alpha+a.x;
   e1.y:= a.y;
   e1:= csqrt(e1);
   e2.x:= gamma*z*z-1.0-alpha-a.x;
   e2.y:= -a.y;
   e2:=csqrt(e2);
   e1:= e1*e2;
   r.x:= (2.0*(1-kappa)*z*z-gamma*z*z+alpha)*(1-z*z)- 0.5*e1.x;
   r.y:= -0.5*e1.y;
   end;
{if n>maxit then
   writeln( 'Error in Rayleigh_Pole function: too many iterations');}
rayleigh_pole:= z;
end;


function axssincosfunc(p: pointer; zeta: double): double;
var
data: ^taxskerndata;
result: double;
begin
data:=p;
if data^.r<=0.0 then
   begin
   axssincosfunc:=0.0;
   exit;
   end;

result:= axskernfunc(p,zeta);
axssincosfunc:= result*sqrt(1.0/pi/data^.delta/zeta/data^.r);
end;

function axscosfunc(p: pointer; zeta: double): double;
var
data: ^taxskerndata;
result: double;
begin
data:=p;
if data^.r<=0.0 then
   begin
   axscosfunc:=0.0;
   exit;
   end;

result:= axskernfunc(p,zeta);
axscosfunc:= result*(1.0/pi/data^.delta/zeta/data^.r);
end;


function axsgreenfunc(mat: mat_const;
         r, z, h, loadr, loadh, omega: double;
         bvptype, loadtype, component, part: integer): double;
const
bzero: array[1..10] of double = (
2.40482555769577276862163187933, 
      5.52007811028631064959660411416,
      8.65372791291101221695419871266, 
      11.7915344390142816137430449119,
      14.9309177084877859477625939974, 
      18.0710639679109225431478829756,
      21.2116366298792589590783933505, 
      24.3524715307493027370579447632, 
      27.4934791320402547958772847, 
      30.63460646843197511754957761);
      
var
a, a1, aux, abserr, b, epsabs, epsrel, resultado, result1,
ressin, rescos: double;
ier, last, leniw, lenw, limit, limlst, lst, maxp1, neval: integer;
work: array[1..1025] of double;
iwork: array[1..250] of integer;
workptr: ^nvec;
iworkptr: ^nintvec;

data: taxskerndata;
spoint: array [1..4] of double;
i,j,k: integer;

begin
with mat do  with data do
   begin
   alpha:= (c33+damp*c33*c_i)/c44;
   beta:=  (c11+damp*c11*c_i)/c44;
   kappa:= ((c13+damp*c13*c_i)+c44)/c44;
   tau:= (c11*(1.0+damp*c_i)-c12*(1.0+damp*c_i))/2.0/c44;
   gamma:= 1.0 + alpha*beta - kappa*kappa;
   delta:= sqrt(rho/c44)*omega;
   end;

data.h:= h;
if (bvptype=loadinfullspace) or (bvptype=loadoverhalfspace) then
   data.h:=0.0;
data.z:= z;
data.r:= r;
data.loadradius:= loadr;
data.loadheight:= loadh;
data.bvptype:= bvptype;
data.loadtype:= loadtype;
data.component:= component;
data.part:= part;
data.mode:= normalmode;

spoint[1]:= 1.0/sqrt(data.beta.x);
spoint[2]:= 1.0;
with data do
   spoint[3]:= rayleigh_pole(alpha.x, beta.x, kappa.x);
for j:=1 to 2 do for k:=1 to 2 do
   if spoint[k]>spoint[k+1] then
      begin
      a:= spoint[k];
      spoint[k]:= spoint[k+1];
      spoint[k+1]:=a;
      end;
spoint[4]:= spoint[3]+2.0;

epsrel:= {delta_load_precision}1.0E-6;
epsabs:=0.0;
limit:= 250;

leniw:= limit;
lenw:= limit*4;
workptr:=@work[1];
iworkptr:= @iwork[1];

a:=0.0;
resultado:= 0.0;
i:=1;
for j:=1 to 10 do
   begin
   b:=bzero[j];
   if r>0.0 then b:= b/data.delta/r;
   while (i<=4) and (spoint[i]<b) do
      begin
      dqags(@axskernfunc,@data, a, spoint[i], epsabs, epsrel, result1, abserr,
        neval, ier, leniw, lenw, last, iworkptr^, workptr^);
      resultado:= resultado + result1;
      a:= spoint[i];
      i:= i+1;
      end;

   dqags(@axskernfunc,@data, a, b, epsabs, epsrel, result1, abserr,
        neval, ier, leniw, lenw, last, iworkptr^, workptr^);
   resultado:= resultado + result1;
   a:=b;
   end;
   

   epsrel:= delta_load_precision;
   epsabs:=0.0;
   limlst:= 50;
   limit:= 100;
   leniw:= limit*2+limlst;
   maxp1:= 21;
   lenw:= leniw*2+maxp1*25;
   workptr:=@work[1];
   iworkptr:= @iwork[1];
   epsabs:= epsrel;
   
   if (loadtype=cylinderload) and (abs(z-data.h)<1.0E-8) and
   (abs(loadr-r)<1.0E-8) and (component=srrz) then
      begin
      data.mode:= wobessel_b;
      a1:= 2.0*data.delta*r;
      result1:=0.0;
      for j:= i to 4 do if spoint[j]>a then
	 begin
	 b:= spoint[j];
	 dqawo(@axscosfunc,@data, a, b, a1, 1, epsabs, epsrel, rescos, abserr,
              neval, ier, leniw, maxp1, lenw, last, iworkptr^, workptr^);
	 result1:= result1-rescos;
	 a:=b;
	 end;

      dqawf(@axscosfunc,@data, b, a1, 1, epsrel,
         rescos, abserr, neval, ier,
        limlst, lst, leniw, maxp1, lenw, iworkptr^, workptr^);
      result1:= result1-rescos;
      end

   else
      begin
      data.mode:= wobessel;
      a1:= data.delta*r;
      ressin:=0.0;
      rescos:=0.0;
      aux:=a;
      for j:= i to 4 do if spoint[j]>a then
	 begin
	 b:= spoint[j];
	 dqawo(@axssincosfunc,@data, a, b, a1, 2, epsabs, epsrel, result1, abserr,
              neval, ier, leniw, maxp1, lenw, last, iworkptr^, workptr^);
	 ressin:= ressin+result1;
	 dqawo(@axssincosfunc,@data, a, b, a1, 1, epsabs, epsrel, result1, abserr,
              neval, ier, leniw, maxp1, lenw, last, iworkptr^, workptr^);
	 rescos:= rescos+ result1;
	 a:=b;
	 end;

      dqawf(@axssincosfunc,@data, b, a1, 2, epsrel,
            result1, abserr, neval, ier,
           limlst, lst, leniw, maxp1, lenw, iworkptr^, workptr^);
      ressin:= ressin + result1;
      dqawf(@axssincosfunc,@data, b, a1, 1, epsrel,
            result1, abserr, neval, ier,
           limlst, lst, leniw, maxp1, lenw, iworkptr^, workptr^);
      rescos:= rescos + result1;
      case data.component of
      uzz,uzr,szzz,szzr: result1:= ressin+rescos;
      urz,urr,srzz,srzr, specialsrzz: result1:= ressin-rescos;
      srrz,srrr:
         begin
         result1:= ressin+rescos;
         data.mode:= wobessel_b;
	 a:= aux;
	 for j:= i to 4 do if spoint[j]>a then
	    begin
	    b:= spoint[j];
	    dqawo(@axssincosfunc,@data, a, b, a1, 2, epsabs, epsrel, ressin, abserr,
        	 neval, ier, leniw, maxp1, lenw, last, iworkptr^, workptr^);
	    dqawo(@axssincosfunc,@data, a, b, a1, 1, epsabs, epsrel, rescos, abserr,
        	 neval, ier, leniw, maxp1, lenw, last, iworkptr^, workptr^);
            result1:= result1 + ressin-rescos;
	    a:=b;
	    end;


         dqawf(@axssincosfunc,@data, b, a, 2, epsabs,
               ressin, abserr, neval, ier,
              limlst, lst, leniw, maxp1, lenw, iworkptr^, workptr^);
         dqawf(@axssincosfunc,@data, b, a, 1, epsrel,
               rescos, abserr, neval, ier,
              limlst, lst, leniw, maxp1, lenw, iworkptr^, workptr^);
         result1:= result1 + ressin-rescos;
         end;
      end{case};
      end;{else}

resultado:= resultado+result1;

case component of
uzz,urz,uzr,urr:
   resultado:= resultado*data.delta/2.0/mat.c44;
else
   resultado:= resultado*sqr(data.delta)/2.0;
end{case};

if (bvptype=loadinhalfspace) and (abs(z-h)<0.0001) and
   ((component=szzz) or (component=srzr)) and (part=realpart) then
   case loadtype of
      pointload: if r=0.0 then resultado:= resultado -1000.0;
      ringload : if r=loadr then  resultado:= resultado -1000.0;
      diskload : if r<loadr then  resultado:= resultado -0.5;
      anularload : if (r<loadr+loadh) and (r>loadr-loadh) then  
              resultado:= resultado -0.5;
      end;{case}

if (bvptype=loadinfullspace) and (abs(z)<0.0001) and
   ((component=szzz) or (component=srzr)) and (part=realpart) then
   case loadtype of
      pointload: if r=0.0 then resultado:=  -1000.0 else resultado:=0.0;
      ringload : if r=loadr then  resultado:=  -1000.0 else resultado:=0.0;
      diskload : if r<loadr then  resultado:=  -0.5 else resultado:=0.0;
      anularload : if (r<loadr+loadh) and (r>loadr-loadh) then
              resultado:=  -0.5 else resultado:=0.0;
      end;{case}

if (bvptype=loadinhalfspace) and (abs(z-h)<loadh) and
   ((component=srzz) or (component=srrr)) and (part=realpart) then
   case loadtype of
      cylinderload : if r=loadr then  resultado:= resultado -0.5;
      end;{case}

if (bvptype=loadinfullspace) and (abs(z)<loadh) and
   ((component=srzz) or (component=srrr)) and (part=realpart) then
   case loadtype of
      cylinderload: if r=loadr then resultado:=  resultado-0.5;
      end;{case}

   axsgreenfunc:= resultado;
end;

function axsgreenfunc(mat: mat_const;
         r, z, h, loadr, loadh, omega: double;
         bvptype, loadtype, component:integer): tcomplex;

var
result: tcomplex;
begin
result.x:= axsgreenfunc(mat, r, z, h, loadr, loadh, omega,
         bvptype, loadtype, component, realpart);
result.y:= axsgreenfunc(mat, r, z, h, loadr, loadh, omega,
         bvptype, loadtype, component, imagpart);
axsgreenfunc:= result;	 
end;


function axskernsrzzpv(p: pointer; zeta: double): double;
var
data: ^taxskerndata;
result: double;
begin
data:=p;
result:= 0.0;
if zeta<>1.0 then result:= axskernsrzz0(p,zeta);
axskernsrzzpv:= result*(zeta-1.0);
end;


function axsgreensrzz0(mat: mat_const;
         r, z, h, loadr, loadh, omega: double;
         bvptype, loadtype, component, part: integer): double;

const
off= 0.1;
var
a, abserr, b, b1, epsabs, epsrel, resultado, result1: double;
ier, last, leniw, lenw, limit, limlst, lst, maxp1, neval: integer;
work: array[1..1025] of double;
iwork: array[1..250] of integer;
workptr: ^nvec;
iworkptr: ^nintvec;

data: taxskerndata;
spoint: array [0..5] of double;
j,k: integer;

j0, j1: double;
k0, k1, k2, sl: tcomplex;

begin
with mat do  with data do
   begin
   alpha:= (c33+damp*c33*c_i)/c44;
   beta:=  (c11+damp*c11*c_i)/c44;
   kappa:= ((c13+damp*c13*c_i)+c44)/c44;
   tau:= (c11*(1.0+damp*c_i)-c12*(1.0+damp*c_i))/2.0/c44;
   gamma:= 1.0 + alpha*beta - kappa*kappa;
   delta:= sqrt(rho/c44)*omega;
   end;

data.h:= h;
data.h:=0.0;
data.z:= z;
data.r:= r;
data.loadradius:= loadr;
data.loadheight:= loadh;
data.bvptype:= bvptype;
data.loadtype:= loadtype;
data.component:= component;
data.part:= part;
data.mode:= normalmode;

j0:= besselj0(loadr*data.delta);
j1:= besselj1(loadr*data.delta);
{besselk012(c_i*loadr*data.delta, k0, k1, k2);}
kzeone(0.0, loadr*data.delta, k0.x, k0.y, k1.x, k1.y);
sl:= c_i*loadr*data.delta*j0*k1;

if loadtype= longcylinderload then
   begin
   if part= realpart then
      axsgreensrzz0:= -sl.x
   else
      axsgreensrzz0:= -sl.y;
   exit;
   end;

spoint[0]:= 0.0;
spoint[1]:= 1.0/sqrt(data.beta.x);
spoint[2]:= 1.0;
with data do
   spoint[3]:= rayleigh_pole(alpha.x, beta.x, kappa.x);
for j:=1 to 2 do for k:=1 to 2 do
   if spoint[k]>spoint[k+1] then
      begin
      a:= spoint[k];
      spoint[k]:= spoint[k+1];
      spoint[k+1]:=a;
      end;
spoint[4]:= spoint[3]+2.0;
if loadr>0.0 then spoint[5]:= 30.0/data.delta/loadr else spoint[5]:= 30.0;
if spoint[4]>spoint[5] then spoint[5]:= spoint[4]+5.0;

{a:=0.0;
b:=spoint[1];}


epsrel:= {delta_load_precision}1.0E-6;
epsabs:=0.0;
limit:= 250;

leniw:= limit;
lenw:= limit*4;
workptr:=@work[1];
iworkptr:= @iwork[1];

resultado:=0.0;
{for k:=0 to 4 do
   begin
   a:= spoint[k];
   b:= spoint[k+1];
   if (b=1.0) and (part=realpart) then
      begin
      b1:= spoint[k+2];
      dqawc(@axskernsrzzpv,@data, a,b1,1.0,epsabs,epsrel, result1, abserr,
             neval,ier, limit,lenw, last, iworkptr^, workptr^);
      end;
   if ((a<>1.0) and (b<>1.0)) or (part=imagpart) then
      dqags(@axskernsrzz0,@data, a, b, epsabs, epsrel, result1, abserr,
            neval, ier, leniw, lenw, last, iworkptr^, workptr^);

   resultado:= resultado+result1;

   end;}
data.mode:= varyimag;
data.aux1:= 0.0;
dqags(@axskernsrzz0, @data, 0.0, off, epsabs, epsrel, result1, abserr,
      neval, ier, leniw, lenw, last, iworkptr^, workptr^);
resultado:= resultado + result1;

data.mode:= normalmode;
data.aux1:= off;
dqags(@axskernsrzz0, @data, 0.0, spoint[5], epsabs, epsrel, result1, abserr,
      neval, ier, leniw, lenw, last, iworkptr^, workptr^);
resultado:= resultado + result1;

data.mode:= varyimag;
data.aux1:= spoint[5];
dqags(@axskernsrzz0, @data, off, 0.0, epsabs, epsrel, result1, abserr,
      neval, ier, leniw, lenw, last, iworkptr^, workptr^);
resultado:= resultado + result1;

   
   epsrel:= delta_load_precision;
   epsabs:=0.0;
   limlst:= 50;
   limit:= 100;
   leniw:= limit*2+limlst;
   maxp1:= 21;
   lenw:= leniw*2+maxp1*25;
   workptr:=@work[1];
   iworkptr:= @iwork[1];
   epsabs:= epsrel;
   
      data.mode:= wobessel_b;
      data.aux1:=0.0;
      a:= 2.0*data.delta*loadr;
      b:= spoint[5];
      dqawf(@axskernsrzz0,@data, b, a, 1, epsabs,
         result1, abserr, neval, ier,
        limlst, lst, leniw, maxp1, lenw, iworkptr^, workptr^);
      


resultado:= (resultado+result1)*data.delta;



{writeln(loadr:10:5,'  ',data.delta:10:5,'  ',j0:10:5);
writeln(k1.x:10:5, '  ', k1.y:10:5);
Writeln('J0= ',j0:10:5);
writeln('K1= ',k1.x:10:5, '  ', k1.y:10:5);
writeln('Res= ', resultado:10:5);
writeln('Sl= ', sl.x:10:5, '  ', sl.y:10:5);}

if part=realpart then
   axsgreensrzz0:= resultado-sl.x
else
   axsgreensrzz0:= resultado-sl.y;
end;

end.
