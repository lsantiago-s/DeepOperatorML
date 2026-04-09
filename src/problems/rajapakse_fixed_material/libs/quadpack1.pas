unit quadpack1;

interface
const
nmax= 64*1024 div sizeof(double) - 8;

type
nvec= array[1..nmax] of double;
nintvec= array[1..nmax] of integer;
vec3 = array[1..3] of double;
vec11= array[1..11] of double;
vec13= array[1..13] of double;
vec25= array[1..25] of double;
vec52= array[1..52] of double;
n25array= array[1..100, 1..25] of double;

integfunc= function(p:pointer; x: double): double;
weightfunc= function(x1, x2, x3, x4, x5: double; i1: integer): double;

{function d1mach(i:integer):double;
function dmax1(c1, c2: double): double;
function dmin1(c1, c2: double): double;}
procedure dgtsl(n: integer; var c,d,e,b:vec25 ; var info: integer);
procedure dqcheb(var x:vec11; var fval: vec25; var cheb12: vec13;
                 var cheb24: vec25);
procedure dqelg(var n: integer; var epstab: vec52; var result,abserr: double;
                var res3la: vec3; var nres: integer);
procedure dqpsrt(var limit,last,maxerr: integer; var ermax: double;
                 var elist: nvec; var iord: nintvec; var nrmax: integer);
procedure dqk21(f: integfunc; p:pointer; a,b: double;
                var result,abserr,resabs,resasc: double);
procedure dqk15i(f: integfunc; p:pointer; boun: double; inf: integer;
                a,b: double; var result,abserr,resabs,resasc: double);
function dqwgtf(x,omega,p2,p3,p4: double; integr: integer):double;
function dqwgtc(x, c, p2, p3, p4: double; kp: integer): double;

procedure dqk15w(f: integfunc; p:pointer; w: weightfunc; p1,p2,p3,p4: double; kp: integer;
                 a,b: double; var result,abserr,resabs,resasc: double);
procedure dqc25f(f: integfunc; p:pointer; a,b,omega: double;
                 integr,nrmom,maxp1,ksave: integer;
                 var result,abserr: double; var neval: integer;
                 var resabs,resasc: double; var momcom: integer;
                 var chebmo: n25array);
procedure dqagie(f: integfunc; p: pointer; bound: double; inf:integer;
                 epsabs,epsrel: double; limit:integer;
                 var result,abserr: double; var neval,ier: integer;
                 var alist,blist,rlist,elist: nvec; var iord: nintvec;
                 var last: integer);
procedure dqagse(f: integfunc; p:pointer; a,b,epsabs,epsrel: double; limit: integer;
                 var result,abserr: double; var neval,ier: integer;
                 var alist,blist,rlist,elist: nvec; var iord: nintvec;
                 var last: integer);
procedure dqawoe (f: integfunc; p:pointer; a,b,omega: double; integr: integer;
                 epsabs,epsrel: double; limit,icall,maxp1: integer;
                 var result,abserr: double; var neval,ier,last:integer;
                 var alist,blist,rlist,elist: nvec; var iord, nnlog: nintvec;
                 var momcom: integer; var chebmo: n25array);
procedure dqawfe(f:integfunc; p:pointer; a,omega: double; integr: integer;
                 epsabs: double; limlst,limit,maxp1: integer;
                 var result,abserr: double; var neval,ier: integer;
                 var rslst,erlst: nvec; var ierlst: nintvec;
                 var lst: integer; var alist,blist,rlist,elist: nvec;
                 var iord,nnlog: nintvec; var chebmo: n25array);
procedure dqawce(f:integfunc; p:pointer; a,b,c,epsabs,epsrel: double;
                 limit: integer; var result,abserr: double;
		 var neval,ier: integer; var alist,blist,rlist,elist: nvec;
		 var iord: nintvec; var last: integer);

procedure dqawf(f: integfunc; p: pointer; a,omega: double; integr: integer;
                epsabs: double; var result,abserr: double;
                var neval,ier: integer; limlst:integer;
                var lst:integer; leniw,maxp1,lenw: integer;
                var iwork: nintvec ; var work: nvec);
procedure dqawo(f: integfunc; p: pointer; a,b,omega: double; integr: integer;
                epsabs,epsrel: double; var result,abserr: double;
                var neval,ier:integer; leniw,maxp1,lenw: integer;
                var last: integer; var iwork: nintvec; var work: nvec);
procedure dqags(f: integfunc; p:pointer; a,b,epsabs,epsrel: double;
                var result,abserr: double; var neval,ier: integer;
                limit,lenw: integer; var last: integer;
                var iwork: nintvec; var work: nvec);

procedure dqagi(f: integfunc; p:pointer; bound: double; inf: integer;
                epsabs,epsrel: double; var result,abserr: double;
                var neval,ier: integer; limit,lenw: integer;
                var last: integer; var iwork: nintvec; var work: nvec);

procedure dqawc(f: integfunc; p:pointer; a,b,c,epsabs,epsrel: double;
                var result,abserr: double; var neval,ier: integer;
                limit,lenw: integer; var last: integer;
		var iwork: nintvec; var work: nvec);

procedure dqc25c(f: integfunc; p: pointer; a,b,c: double; 
                 var result,abserr: double; var krul,neval: integer);


implementation
uses dmach;

{const}
{small: array[1..2] of longint = ( 0,    1048576);
large: array[1..2] of longint = (-1, 2146435071);
right: array[1..2] of longint = ( 0, 1017118720);
diver: array[1..2] of longint = ( 0, 1018167296);
log10: array[1..2] of longint = ( 1352628735, 1070810131);}
{small: array[1..10] of longint= ( 0,    1048576, -1, 2146435071,
                                  0, 1017118720,  0, 1018167296,
                                  1352628735, 1070810131);}



{function d1mach(i:integer):double;
type
resp= array [1..5] of double;

var
b: ^resp;

begin
b:= @small;
d1mach:= b^[i];
end;

function dmax1(c1, c2: double): double;
begin
if c1<c2 then dmax1:=c2 else dmax1:=c1;
end;

function dmin1(c1, c2: double): double;
begin
if c1>c2 then dmin1:=c2 else dmin1:=c1;
end;}

procedure dgtsl(n: integer; var c,d,e,b:vec25 ; var info: integer);
var
k,kb,kp1,nm1,nm2: integer;
t: double;
begin
info := 0;
c[1] := d[1];
nm1 := n - 1;
if (nm1 >= 1) then
   begin
   d[1] := e[1];
   e[1] := 0.0;
   e[n] := 0.0;

   for k := 1 to nm1 do
      begin
      kp1 := k + 1;
      if (abs(c[kp1]) >= abs(c[k])) then
         begin
         t := c[kp1];
         c[kp1] := c[k];
         c[k] := t;
         t := d[kp1];
         d[kp1] := d[k];
         d[k] := t;
         t := e[kp1];
         e[kp1] := e[k];
         e[k] := t;
         t := b[kp1];
         b[kp1] := b[k];
         b[k] := t;
         end;

      if (c[k] = 0.0) then
         begin
         info := k;
         exit;
         end;

      t := -c[kp1]/c[k];
      c[kp1] := d[kp1] + t*d[k];
      d[kp1] := e[kp1] + t*e[k];
      e[kp1] := 0.0;
      b[kp1] := b[kp1] + t*b[k];
      end;
   end;

if (c[n] = 0.0) then
   begin
   info := n;
   exit;
   end;

nm2 := n - 2;
b[n] := b[n]/c[n];
if (n <> 1) then
   begin
   b[nm1] := (b[nm1] - d[nm1]*b[n])/c[nm1];
   if (nm2 >= 1) then
      for  kb := 1 to nm2 do
         begin
         k := nm2 - kb + 1;
         b[k] := (b[k] - d[k]*b[k+1] - e[k]*b[k+2])/c[k];
         end;
   end;
end;

procedure dqcheb(var x:vec11; var fval: vec25; var cheb12: vec13;
                 var cheb24: vec25);

var
  alam,alam1,alam2,part1,part2,part3: double;
  i,j: integer;
  v: array[1..12] of double;

begin
for i:=1 to 12 do
   begin
   j := 26-i;
   v[i] := fval[i]-fval[j];
   fval[i] := fval[i]+fval[j];
   end;

alam1 := v[1]-v[9];
alam2 := x[6]*(v[3]-v[7]-v[11]);
cheb12[4] := alam1+alam2;
cheb12[10] := alam1-alam2;
alam1 := v[2]-v[8]-v[10];
alam2 := v[4]-v[6]-v[12];
alam := x[3]*alam1+x[9]*alam2;
cheb24[4] := cheb12[4]+alam;
cheb24[22] := cheb12[4]-alam;
alam := x[9]*alam1-x[3]*alam2;
cheb24[10] := cheb12[10]+alam;
cheb24[16] := cheb12[10]-alam;
part1 := x[4]*v[5];
part2 := x[8]*v[9];
part3 := x[6]*v[7];
alam1 := v[1]+part1+part2;
alam2 := x[2]*v[3]+part3+x[10]*v[11];
cheb12[2] := alam1+alam2;
cheb12[12] := alam1-alam2;
alam := x[1]*v[2]+x[3]*v[4]+x[5]*v[6]+x[7]*v[8]+x[9]*v[10]+x[11]*v[12];
cheb24[2] := cheb12[2]+alam;
cheb24[24] := cheb12[2]-alam;
alam := x[11]*v[2]-x[9]*v[4]+x[7]*v[6]-x[5]*v[8]+x[3]*v[10]-x[1]*v[12];
cheb24[12] := cheb12[12]+alam;
cheb24[14] := cheb12[12]-alam;
alam1 := v[1]-part1+part2;
alam2 := x[10]*v[3]-part3+x[2]*v[11];
cheb12[6] := alam1+alam2;
cheb12[8] := alam1-alam2;
alam := x[5]*v[2]-x[9]*v[4]-x[1]*v[6]-x[11]*v[8]+x[3]*v[10]+x[7]*v[12];
cheb24[6] := cheb12[6]+alam;
cheb24[20] := cheb12[6]-alam;
alam := x[7]*v[2]-x[3]*v[4]-x[11]*v[6]+x[1]*v[8]-x[9]*v[10]-x[5]*v[12];
cheb24[8] := cheb12[8]+alam;
cheb24[18] := cheb12[8]-alam;
for i:=1 to 6 do
   begin
   j := 14-i;
   v[i] := fval[i]-fval[j];
   fval[i] := fval[i]+fval[j];
   end;
alam1 := v[1]+x[8]*v[5];
alam2 := x[4]*v[3];
cheb12[3] := alam1+alam2;
cheb12[11] := alam1-alam2;
cheb12[7] := v[1]-v[5];
alam := x[2]*v[2]+x[6]*v[4]+x[10]*v[6];
cheb24[3] := cheb12[3]+alam;
cheb24[23] := cheb12[3]-alam;
alam := x[6]*(v[2]-v[4]-v[6]);
cheb24[7] := cheb12[7]+alam;
cheb24[19] := cheb12[7]-alam;
alam := x[10]*v[2]-x[6]*v[4]+x[2]*v[6];
cheb24[11] := cheb12[11]+alam;
cheb24[15] := cheb12[11]-alam;
for i:=1 to 3 do
   begin
   j := 8-i;
   v[i] := fval[i]-fval[j];
   fval[i] := fval[i]+fval[j];
   end;
cheb12[5] := v[1]+x[8]*v[3];
cheb12[9] := fval[1]-x[8]*fval[3];
alam := x[4]*v[2];
cheb24[5] := cheb12[5]+alam;
cheb24[21] := cheb12[5]-alam;
alam := x[8]*fval[2]-fval[4];
cheb24[9] := cheb12[9]+alam;
cheb24[17] := cheb12[9]-alam;
cheb12[1] := fval[1]+fval[3];
alam := fval[2]+fval[4];
cheb24[1] := cheb12[1]+alam;
cheb24[25] := cheb12[1]-alam;
cheb12[13] := v[1]-v[3];
cheb24[13] := cheb12[13];
alam := 1.0/6.0;
for i:=2 to 12 do
   cheb12[i] := cheb12[i]*alam;
alam := 0.5*alam;
cheb12[1] := cheb12[1]*alam;
cheb12[13] := cheb12[13]*alam;
for i:=2 to 24 do
   cheb24[i] := cheb24[i]*alam;
cheb24[1] := 0.5*alam*cheb24[1];
cheb24[25] := 0.5*alam*cheb24[25];

end;

procedure dqelg(var n: integer; var epstab: vec52; var result,abserr: double;
                var res3la: vec3; var nres: integer);

var
dabs,delta1,delta2,delta3,epmach,epsinf,error,err1,err2,err3,
e0,e1,e1abs,e2,e3,oflow,res,ss,tol1,tol2,tol3: double;

i,ib,ib2,ie,indx,k1,k2,k3,limexp,newelm,num: integer;

begin
epmach := d1mach(4);
oflow := d1mach(2);
nres := nres+1;
abserr := oflow;
result := epstab[n];
if n>=3 then
   begin
   limexp := 50;
   epstab[n+2] := epstab[n];
   newelm := (n-1) div 2;
   epstab[n] := oflow;
   num := n;
   k1 := n;
   for i := 1 to newelm do
      begin
      k2 := k1-1;
      k3 := k1-2;
      res := epstab[k1+2];
      e0 := epstab[k3];
      e1 := epstab[k2];
      e2 := res;
      e1abs := abs(e1);
      delta2 := e2-e1;
      err2 := abs(delta2);
      tol2 := dmax1(abs(e2),e1abs)*epmach;
      delta3 := e1-e0;
      err3 := abs(delta3);
      tol3 := dmax1(e1abs,abs(e0))*epmach;

{c
c           if e0, e1 and e2 are equal to within machine
c           accuracy, convergence is assumed.
c           result := e2
c           abserr := abs(e1-e0)+abs(e2-e1)
c}
      if (err2<=tol2) and (err3<=tol3) then
         begin
         result:= res;
         abserr:=err2+err3;
         abserr := dmax1(abserr,0.5E+00*epmach*abs(result));
         exit;
         end;

      e3 := epstab[k1];
      epstab[k1] := e1;
      delta1 := e1-e3;
      err1 := abs(delta1);
      tol1 := dmax1(e1abs,abs(e3))*epmach;
{c
c           if two elements are very close to each other, omit
c           a part of the table by adjusting the value of n
c}
      if (err1<=tol1) or (err2<=tol2) or (err3<=tol3) then{ go to 20}
        begin
        n := i+i-1;
        break;
        end;
      ss := 0.1E+01/delta1+0.1E+01/delta2-0.1E+01/delta3;
      epsinf := abs(ss*e1);
{c
c           test to detect irregular behaviour in the table, and
c           eventually omit a part of the table adjusting the value
c           of n.
c}
      if epsinf<=0.1E-03 then
         begin
         n := i+i-1;
         break;
         end;
{c
c           compute a new element and eventually adjust
c           the value of result.
c}
      res := e1+0.1E+01/ss;
      epstab[k1] := res;
      k1 := k1-2;
      error := err2+abs(res-e2)+err3;
      if error<=abserr then
         begin
         abserr := error;
         result := res;
         end;
      end;

{c
c           shift the table.
c}
   if n=limexp then n := 2*(limexp div 2)-1;
   ib := 1;
   if (num div 2)*2 = num then ib := 2;
   ie := newelm+1;
   for i:=1 to ie do
      begin
      ib2 := ib+2;
      epstab[ib] := epstab[ib2];
      ib := ib2;
      end;
   if num<>n then
      begin
      indx := num-n+1;
      for i := 1 to n do
         begin
         epstab[i]:= epstab[indx];
         indx := indx+1;
         end;
      end;
   if nres>=4 then
{c
c           compute error estimate
c}
      begin
      abserr := abs(result-res3la[3])+abs(result-res3la[2])
                +abs(result-res3la[1]);
      res3la[1] := res3la[2];
      res3la[2] := res3la[3];
      res3la[3] := result;
      end
   else
      begin
      res3la[nres] := result;
      abserr := oflow;
      end;
   end;
abserr := dmax1(abserr,0.5E+00*epmach*abs(result));
end;

procedure dqpsrt(var limit,last,maxerr: integer; var ermax: double;
                 var elist: nvec; var iord: nintvec; var nrmax: integer);

label
lab60, lab80, lab90;

var
   errmax,errmin: double;
   i,ibeg,ido,isucc,j,jbnd,jupbn,k: integer;

begin
{c
c           check whether the list contains more than
c           two error estimates.
c
c***first executable statement  dqpsrt}

if (last<=2) then
   begin
   iord[1] := 1;
   iord[2] := 2;
   maxerr := iord[nrmax];
   ermax := elist[maxerr];
   exit;
   end;

{c
c           this part of the routine is only executed if, due to a
c           difficult integrand, subdivision increased the error
c           estimate. in the normal case the insert procedure should
c           start after the nrmax-th largest error estimate.
c}
errmax := elist[maxerr];
if (nrmax<>1) then
   begin
   ido := nrmax-1;
   for i := 1 to ido do
      begin
      isucc := iord[nrmax-1];
{c ***jump out of do-loop}
      if(errmax<=elist[isucc]) then break;
      iord[nrmax] := isucc;
      nrmax := nrmax-1;
      end;
   end;

{c
c           compute the number of elements in the list to be maintained
c           in descending order. this number depends on the number of
c           subdivisions still allowed.
c}
jupbn := last;
if(last>(limit div 2+2)) then jupbn := limit+3-last;
errmin := elist[last];

{c
c           insert errmax by traversing the list top-down,
c           starting comparison from the element elist(iord(nrmax+1)).
c}
jbnd := jupbn-1;
ibeg := nrmax+1;
if (ibeg>jbnd) then
   begin
   iord[jbnd] := maxerr;
   iord[jupbn] := last;
   maxerr := iord[nrmax];
   ermax := elist[maxerr];
   exit;
   end;

for i:=ibeg to jbnd do
   begin
   isucc := iord[i];
   {c ***jump out of do-loop}
   if(errmax>=elist[isucc]) then goto lab60;
   iord[i-1] := isucc;
   end;
iord[jbnd] := maxerr;
iord[jupbn] := last;
maxerr := iord[nrmax];
ermax := elist[maxerr];
exit;

{c
c           insert errmin by traversing the list bottom-up.
c}
lab60: iord[i-1] := maxerr;
k := jbnd;
for j:=i to jbnd do
   begin
   isucc := iord[k];
   {c ***jump out of do-loop}
   if (errmin<elist[isucc]) then goto lab80;
   iord[k+1] := isucc;
   k := k-1;
   end;
iord[i] := last;
goto lab90;
lab80: iord[k+1] := last;

{c
c           set maxerr and ermax.
c}
lab90: maxerr := iord[nrmax];
ermax := elist[maxerr];
end;

procedure dqk21(f: integfunc; p:pointer; a,b: double;
                var result,abserr,resabs,resasc: double);

const
wg: array[1..5] of double = (
 0.066671344308688137593568809893332,
 0.149451349150580593145776339657697,
 0.219086362515982043995534934228163,
 0.269266719309996355091226921569469,
 0.295524224714752870173892994651338 );
xgk: array[1..11] of double = (
 0.995657163025808080735527280689003,
 0.973906528517171720077964012084452,
 0.930157491355708226001207180059508,
 0.865063366688984510732096688423493,
 0.780817726586416897063717578345042,
 0.679409568299024406234327365114874,
 0.562757134668604683339000099272694,
 0.433395394129247190799265943165784,
 0.294392862701460198131126603103866,
 0.148874338981631210884826001129720,
 0.000000000000000000000000000000000 );
wgk: array[1..11] of double = (
 0.011694638867371874278064396062192,
 0.032558162307964727478818972459390,
 0.054755896574351996031381300244580,
 0.075039674810919952767043140916190,
 0.093125454583697605535065465083366,
 0.109387158802297641899210590325805,
 0.123491976262065851077958109831074,
 0.134709217311473325928054001771707,
 0.142775938577060080797094273138717,
 0.147739104901338491374841515972068,
 0.149445554002916905664936468389821 );

var
absc,centr,dabs,dhlgth,epmach,fc,fsum,fval1,fval2,hlgth,
resg,resk,reskh,uflow: double;
j,jtw,jtwm1 : integer;
fv1,fv2: array [1..10] of double;

begin
epmach := d1mach(4);
uflow := d1mach(1);

centr := 0.5*(a+b);
hlgth := 0.5*(b-a);
dhlgth := abs(hlgth);
{c
c           compute the 21-point kronrod approximation to
c           the integral, and estimate the absolute error.
c}
resg := 0.0;
fc := f(p,centr);
resk := wgk[11]*fc;
resabs := abs(resk);
for j:=1 to 5 do
   begin
   jtw := 2*j;
   absc := hlgth*xgk[jtw];
   fval1 := f(p,centr-absc);
   fval2 := f(p,centr+absc);
   fv1[jtw] := fval1;
   fv2[jtw] := fval2;
   fsum := fval1+fval2;
   resg := resg+wg[j]*fsum;
   resk := resk+wgk[jtw]*fsum;
   resabs := resabs+wgk[jtw]*(abs(fval1)+abs(fval2));
   end;
for j := 1 to 5 do
   begin
   jtwm1 := 2*j-1;
   absc := hlgth*xgk[jtwm1];
   fval1 := f(p,centr-absc);
   fval2 := f(p,centr+absc);
   fv1[jtwm1] := fval1;
   fv2[jtwm1] := fval2;
   fsum := fval1+fval2;
   resk := resk+wgk[jtwm1]*fsum;
   resabs := resabs+wgk[jtwm1]*(abs(fval1)+abs(fval2));
   end;
reskh := resk*0.5;
resasc := wgk[11]*abs(fc-reskh);
for j:=1 to 10 do
   resasc := resasc+wgk[j]*(abs(fv1[j]-reskh)+abs(fv2[j]-reskh));
result := resk*hlgth;
resabs := resabs*dhlgth;
resasc := resasc*dhlgth;
abserr := abs((resk-resg)*hlgth);
if(resasc<>0.0) and (abserr<>0.0) then
   abserr := resasc*dmin1(0.1E+01,exp(ln(0.2E+03*abserr/resasc)*1.5));
if(resabs>uflow/(0.5E+02*epmach)) then
   abserr := dmax1((epmach*0.5E+02)*resabs,abserr);
end;


procedure dqk15i(f: integfunc; p: pointer; boun: double; inf: integer;
                a,b: double; var result,abserr,resabs,resasc: double);
var
absc,absc1,absc2,centr,dabs,dinf,epmach,fc,fsum,fval1,fval2,
hlgth,resg,resk,reskh,tabsc1,tabsc2,uflow: double;
j: integer;
fv1,fv2: array[1..7] of double;

{c
c           the abscissae and weights are supplied for the interval
c           (-1,1).  because of symmetry only the positive abscissae and
c           their corresponding weights are given.
c
c           xgk    - abscissae of the 15-point kronrod rule
c                    xgk(2), xgk(4), ... abscissae of the 7-point
c                    gauss rule
c                    xgk(1), xgk(3), ...  abscissae which are optimally
c                    added to the 7-point gauss rule
c
c           wgk    - weights of the 15-point kronrod rule
c
c           wg     - weights of the 7-point gauss rule, corresponding
c                    to the abscissae xgk(2), xgk(4), ...
c                    wg(1), wg(3), ... are set to zero.
c}
const
wg: array[1..8] of double=( 0.0, 0.129484966168869693270611432679082,
                            0.0, 0.279705391489276667901467771423780,
                            0.0, 0.381830050505118944950369775488975,
                            0.0, 0.417959183673469387755102040816327);

xgk:array[1..8] of double=( 0.991455371120812639206854697526329,
                            0.949107912342758524526189684047851,
                            0.864864423359769072789712788640926,
                            0.741531185599394439863864773280788,
                            0.586087235467691130294144838258730,
                            0.405845151377397166906606412076961,
                            0.207784955007898467600689403773245,
                            0.000000000000000000000000000000000);

wgk:array[1..8] of double=( 0.022935322010529224963732008058970,
                            0.063092092629978553290700663189204,
                            0.104790010322250183839876322541518,
                            0.140653259715525918745189590510238,
                            0.169004726639267902826583426598550,
                            0.190350578064785409913256402421014,
                            0.204432940075298892414161999234649,
                            0.209482141084727828012999174891714);


begin
epmach := d1mach(4);
uflow := d1mach(1);
dinf := 1.0;
if dinf>inf then dinf:=inf;

centr := 0.5*(a+b);
hlgth := 0.50*(b-a);
tabsc1 := boun+dinf*(0.1E+01-centr)/centr;
fval1 := f(p,tabsc1);
if(inf=2) then fval1 := fval1+f(p,-tabsc1);
fc := (fval1/centr)/centr;
{
c           compute the 15-point kronrod approximation to
c           the integral, and estimate the error.
c}
resg := wg[8]*fc;
resk := wgk[8]*fc;
resabs := abs(resk);
for j:=1 to 7 do
   begin
   absc := hlgth*xgk[j];
   absc1 := centr-absc;
   absc2 := centr+absc;
   tabsc1 := boun+dinf*(0.1E+01-absc1)/absc1;
   tabsc2 := boun+dinf*(0.1E+01-absc2)/absc2;
   fval1 := f(p,tabsc1);
   fval2 := f(p,tabsc2);
   if(inf=2) then fval1 := fval1+f(p,-tabsc1);
   if(inf=2) then fval2 := fval2+f(p,-tabsc2);
   fval1 := (fval1/absc1)/absc1;
   fval2 := (fval2/absc2)/absc2;
   fv1[j] := fval1;
   fv2[j] := fval2;
   fsum := fval1+fval2;
   resg := resg+wg[j]*fsum;
   resk := resk+wgk[j]*fsum;
   resabs := resabs+wgk[j]*(abs(fval1)+abs(fval2));
   end;
reskh := resk*0.5;
resasc := wgk[8]*abs(fc-reskh);
for j:=1 to 7 do
   resasc := resasc+wgk[j]*(abs(fv1[j]-reskh)+abs(fv2[j]-reskh));
result := resk*hlgth;
resasc := resasc*hlgth;
resabs := resabs*hlgth;
abserr := abs((resk-resg)*hlgth);
if(resasc<>0.0) and (abserr<>0.0) then
   abserr := resasc*dmin1(0.1E+01,
             exp(1.5*ln(0.2E+03*abserr/resasc)));
if(resabs > uflow/(0.5E+02*epmach)) then
    abserr := dmax1((epmach*0.5E+02)*resabs,abserr);
end;


function dqwgtf(x,omega,p2,p3,p4: double; integr: integer):double;
begin
if integr=1 then
   dqwgtf:= cos(omega*x)
else
   dqwgtf:= sin(omega*x);
end;


procedure dqk15w(f: integfunc; p:pointer; w: weightfunc; p1,p2,p3,p4: double; kp: integer;
                 a,b: double; var result,abserr,resabs,resasc: double);
const
xgk: array[1..8] of double=(0.9914553711208126, 0.9491079123427585,
                            0.8648644233597691, 0.7415311855993944,
                            0.5860872354676911, 0.4058451513773972,
                            0.2077849550078985, 0.0000000000000000);

wgk: array[1..8] of double=(0.2293532201052922E-01, 0.6309209262997855E-01,
                            0.1047900103222502E+00, 0.1406532597155259E+00,
                            0.1690047266392679E+00, 0.1903505780647854E+00,
                            0.2044329400752989E+00, 0.2094821410847278E+00);

wg: array[1..4] of double=(0.1294849661688697E+00, 0.2797053914892767E+00,
                           0.3818300505051889E+00, 0.4179591836734694E+00);

var
absc,absc1,absc2,centr,dhlgth,epmach,fc,fsum,fval1,fval2,
   hlgth,resg,resk,reskh,uflow: double;
j,jtw,jtwm1: integer;
fv1,fv2: array[1..7] of double;

begin
epmach := d1mach(4);
uflow := d1mach(1);

centr := 0.5*(a+b);
hlgth := 0.5*(b-a);
dhlgth := abs(hlgth);

{c           compute the 15-point kronrod approximation to the
c           integral, and estimate the error.
c}
fc := f(p,centr)*w(centr,p1,p2,p3,p4,kp);
resg := wg[4]*fc;
resk := wgk[8]*fc;
resabs := abs(resk);
for j:=1 to 3 do
        begin
        jtw := j*2;
        absc := hlgth*xgk[jtw];
        absc1 := centr-absc;
        absc2 := centr+absc;
        fval1 := f(p,absc1)*w(absc1,p1,p2,p3,p4,kp);
        fval2 := f(p,absc2)*w(absc2,p1,p2,p3,p4,kp);
        fv1[jtw] := fval1;
        fv2[jtw] := fval2;
        fsum := fval1+fval2;
        resg := resg+wg[j]*fsum;
        resk := resk+wgk[jtw]*fsum;
        resabs := resabs+wgk[jtw]*(abs(fval1)+abs(fval2));
        end;
for j:=1 to 4 do
        begin
        jtwm1 := j*2-1;
        absc := hlgth*xgk[jtwm1];
        absc1 := centr-absc;
        absc2 := centr+absc;
        fval1 := f(p,absc1)*w(absc1,p1,p2,p3,p4,kp);
        fval2 := f(p,absc2)*w(absc2,p1,p2,p3,p4,kp);
        fv1[jtwm1] := fval1;
        fv2[jtwm1] := fval2;
        fsum := fval1+fval2;
        resk := resk+wgk[jtwm1]*fsum;
        resabs := resabs+wgk[jtwm1]*(abs(fval1)+abs(fval2));
        end;
reskh := resk*0.5;
resasc := wgk[8]*abs(fc-reskh);
for j:=1 to 7 do
        resasc := resasc+wgk[j]*(abs(fv1[j]-reskh)+abs(fv2[j]-reskh));
result := resk*hlgth;
resabs := resabs*dhlgth;
resasc := resasc*dhlgth;
abserr := abs((resk-resg)*hlgth);
if (resasc<>0.0) and (abserr<>0.0) then
       abserr := resasc*dmin1(1.0,exp(ln(0.2E+03*abserr/resasc)*1.5E+00));
if (resabs>uflow/(0.5E+02*epmach)) then
      abserr := dmax1((epmach*0.5E+02)*resabs,abserr)
end;

procedure dqc25f(f: integfunc; p: pointer; a,b,omega: double;
                 integr,nrmom,maxp1,ksave: integer;
                 var result,abserr: double; var neval: integer;
                 var resabs,resasc: double; var momcom: integer;
                 var chebmo: n25array);
const
{c           the vector x contains the values cos(k*pi/24)
c           k = 1, ...,11, to be used for the chebyshev expansion of f
c}
x: vec11 =(0.991444861373810411144557526928563,
                           0.965925826289068286749743199728897,
                           0.923879532511286756128183189396788,
                           0.866025403784438646763723170752936,
                           0.793353340291235164579776961501299,
                           0.707106781186547524400844362104849,
                           0.608761429008720639416097542898164,
                           0.500000000000000000000000000000000,
                           0.382683432365089771728459984030399,
                           0.258819045102520762348898837624048,
                           0.130526192220051591548406227895489);

var
ac,an,an2,as1,asap,ass,centr,conc,cons,cospar,
  estc,ests,hlgth,oflow,parint,par2,par22,
  p2,p3,p4,resc12,resc24,ress12,ress24,sinpar: double;

i,iers,isym,j,k,m,noequ,noeq1: integer;

cheb12: vec13;
cheb24,d,d1,d2,fval: vec25;
v: array[1..28] of double;
vecptr: ^vec25;

begin
oflow := d1mach(2);
centr := 0.5*(b+a);
hlgth := 0.5*(b-a);
parint := omega*hlgth;
{c
c           compute the integral using the 15-point gauss-kronrod
c           formula if the value of the parameter in the integrand
c           is small.
c}
if(abs(parint)<=0.2E+01) then
      begin
      dqk15w(f,p,@dqwgtf,omega,p2,p3,p4,integr,a,b,result,abserr,resabs,resasc);
      neval := 15;
      exit;
      end;

{c
c           compute the integral using the generalized clenshaw-
c           curtis method.
c}
conc := hlgth*cos(centr*omega);
cons := hlgth*sin(centr*omega);
resasc := oflow;
neval := 25;
{c
c           check whether the chebyshev moments for this interval
c           have already been computed.
c}
if (nrmom>=momcom) and (ksave<>1) then
   begin
{c
c           compute a new set of chebyshev moments.
c}
   m := momcom+1;
   par2 := parint*parint;
   par22 := par2+0.2E+01;
   sinpar := sin(parint);
   cospar := cos(parint);
{c
c           compute the chebyshev moments with respect to cosine.
c}
   v[1] := 0.2E+01*sinpar/parint;
   v[2] := (0.8E+01*cospar+(par2+par2-0.8E+01)*sinpar/parint)/par2;
   v[3] := (0.32E+02*(par2-0.12E+02)*cospar+(0.2E+01*
        ((par2-0.80E+02)*par2+0.192E+03)*sinpar)/parint)/(par2*par2);
   ac := 0.8E+01*cospar;
   as1 := 0.24E+02*parint*sinpar;
   if (abs(parint)<=0.24E+02) then
      begin
{c
c           compute the chebyshev moments as the solutions of a
c           boundary value problem with 1 initial value (v(3)) and 1
c           end value (computed using an asymptotic formula).
c}
      noequ := 25;
      noeq1 := noequ-1;
      an := 0.6E+01;
      for k := 1 to noeq1 do
         begin
         an2 := an*an;
         d[k] := -0.2E+01*(an2-0.4E+01)*(par22-an2-an2);
         d2[k] := (an-0.1E+01)*(an-0.2E+01)*par2;
         d1[k+1] := (an+0.3E+01)*(an+0.4E+01)*par2;
         v[k+3] := as1-(an2-0.4E+01)*ac;
         an := an+0.2E+01;
         end;
      an2 := an*an;
      d[noequ] := -0.2E+01*(an2-0.4E+01)*(par22-an2-an2);
      v[noequ+3] := as1-(an2-0.4E+01)*ac;
      v[4] := v[4]-0.56E+02*par2*v[3];
      ass := parint*sinpar;
      asap := (((((0.210E+03*par2-0.1E+01)*cospar-(0.105E+03*par2
             -0.63E+02)*ass)/an2-(0.1E+01-0.15E+02*par2)*cospar
             +0.15E+02*ass)/an2-cospar+0.3E+01*ass)/an2-cospar)/an2;
      v[noequ+3] := v[noequ+3]-0.2E+01*asap*par2*(an-0.1E+01)*(an-0.2E+01);
{c
c           solve the tridiagonal system by means of gaussian
c           elimination with partial pivoting.
c
c***        call to dgtsl must be replaced by call to
c***        double precision version of linpack routine sgtsl
c}
      vecptr:=@v[4];
      dgtsl(noequ,d1,d,d2,vecptr^,iers);
      end
   else
      begin
{c
c           compute the chebyshev moments by means of forward
c           recursion.
c}
      an := 0.4E+01;
      for i := 4 to 13 do
         begin
         an2 := an*an;
         v[i] := ((an2-0.4E+01)*(0.2E+01*(par22-an2-an2)*v[i-1]-ac)
               +as1-par2*(an+0.1E+01)*(an+0.2E+01)*v[i-2])/
                (par2*(an-0.1E+01)*(an-0.2E+01));
         an := an+0.2E+01;
         end;
      end;

   for j := 1 to 13 do
        chebmo[m,2*j-1] := v[j];
{c
c           compute the chebyshev moments with respect to sine.
c}
   v[1] := 0.2E+01*(sinpar-parint*cospar)/par2;
   v[2] := (0.18E+02-0.48E+02/par2)*sinpar/par2
          +(-0.2E+01+0.48E+02/par2)*cospar/parint;
   ac := -0.24E+02*parint*cospar;
   as1 := -0.8E+01*sinpar;
   if(abs(parint)<=0.24E+02) then
      begin
{c
c           compute the chebyshev moments as the solutions of a boundary
c           value problem with 1 initial value (v(2)) and 1 end value
c           (computed using an asymptotic formula).
c}
      an := 0.5E+01;
      for k := 1 to noeq1 do
         begin
         an2 := an*an;
         d[k] := -0.2E+01*(an2-0.4E+01)*(par22-an2-an2);
         d2[k] := (an-0.1E+01)*(an-0.2E+01)*par2;
         d1[k+1] := (an+0.3E+01)*(an+0.4E+01)*par2;
         v[k+2] := ac+(an2-0.4E+01)*as1;
         an := an+0.2E+01;
         end;
      an2 := an*an;
      d[noequ] := -0.2E+01*(an2-0.4E+01)*(par22-an2-an2);
      v[noequ+2] := ac+(an2-0.4E+01)*as1;
      v[3] := v[3]-0.42E+02*par2*v[2];
      ass := parint*cospar;
      asap := (((((0.105E+03*par2-0.63E+02)*ass+(0.210E+03*par2
            -0.1E+01)*sinpar)/an2+(0.15E+02*par2-0.1E+01)*sinpar-
             0.15E+02*ass)/an2-0.3E+01*ass-sinpar)/an2-sinpar)/an2;
      v[noequ+2] := v[noequ+2]-0.2E+01*asap*par2*(an-0.1E+01)
                    *(an-0.2E+01);
{c
c           solve the tridiagonal system by means of gaussian
c           elimination with partial pivoting.
c
c***        call to dgtsl must be replaced by call to
c***        double precision version of linpack routine sgtsl
c}
      vecptr:=@v[3];
      dgtsl(noequ,d1,d,d2,vecptr^,iers);
      end
   else
{c
c           compute the chebyshev moments by means of forward recursion.
c}
      begin
      an := 0.3E+01;
      for i := 3 to 12 do
         begin
         an2 := an*an;
         v[i] := ((an2-0.4E+01)*(0.2E+01*(par22-an2-an2)*v[i-1]+as1)
                 +ac-par2*(an+0.1E+01)*(an+0.2E+01)*v[i-2])
                 /(par2*(an-0.1E+01)*(an-0.2E+01));
         an := an+0.2E+01;
         end;
      end;
   for j := 1 to 12 do
        chebmo[m,2*j] := v[j];
   end;

if (nrmom<momcom) then m := nrmom+1;
if (momcom<(maxp1-1)) and (nrmom>=momcom) then momcom := momcom+1;
{c
c           compute the coefficients of the chebyshev expansions
c           of degrees 12 and 24 of the function f.
c}
fval[1] := 0.5E+00*f(p,centr+hlgth);
fval[13] := f(p,centr);
fval[25] := 0.5E+00*f(p,centr-hlgth);
for i := 2 to 12 do
   begin
   isym := 26-i;
   fval[i] := f(p,hlgth*x[i-1]+centr);
   fval[isym] := f(p,centr-hlgth*x[i-1]);
   end;
dqcheb(x,fval,cheb12,cheb24);
{c
c           compute the integral and error estimates.
c}
resc12 := cheb12[13]*chebmo[m,13];
ress12 := 0.0;
k := 11;
for j := 1 to 6 do
   begin
   resc12 := resc12+cheb12[k]*chebmo[m,k];
   ress12 := ress12+cheb12[k+1]*chebmo[m,k+1];
   k := k-2;
   end;
resc24 := cheb24[25]*chebmo[m,25];
ress24 := 0.0;
resabs := abs(cheb24[25]);
k := 23;
for j := 1 to 12 do
   begin
   resc24 := resc24+cheb24[k]*chebmo[m,k];
   ress24 := ress24+cheb24[k+1]*chebmo[m,k+1];
   resabs := abs(cheb24[k])+abs(cheb24[k+1]);
   k := k-2;
   end;
estc := abs(resc24-resc12);
ests := abs(ress24-ress12);
resabs := resabs*abs(hlgth);
if (integr<>2) then
   begin
   result := conc*resc24-cons*ress24;
   abserr := abs(conc*estc)+abs(cons*ests);
   end
else
   begin
   result := conc*ress24+cons*resc24;
   abserr := abs(conc*ests)+abs(cons*estc);
   end;
end;

procedure dqagie(f: integfunc; p:pointer; bound: double; inf:integer;
                 epsabs,epsrel: double; limit:integer;
                 var result,abserr: double; var neval,ier: integer;
                 var alist,blist,rlist,elist: nvec; var iord: nintvec;
                 var last: integer);

var
abseps,area,area1,area12,area2,a1,
a2,boun,b1,b2,correc,defabs,defab1,defab2,
dres,epmach,erlarg,erlast,
errbnd,errmax,error1,error2,erro12,errsum,ertest,oflow,resabs,
reseps,small,uflow: double;

id,ierro,iroff1,iroff2,iroff3,jupbnd,k,ksgn,
ktmin,maxerr,nres,nrmax,numrl2, ilast: integer;

extrap,noext: boolean;

res3la: vec3;
rlist2: vec52;

label
lab90, lab100, lab105, lab110, lab115, lab130;

begin
epmach := d1mach(4);
{c
c           test on validity of parameters
c           -----------------------------
c}
ier := 0;
neval := 0;
last := 0;
result := 0.0;
abserr := 0.0;
alist[1] := 0.0;
blist[1] := 0.1E+01;
rlist[1] := 0.0;
elist[1] := 0.0;
iord[1] := 0;
if(epsabs<=0.0) and (epsrel<dmax1(0.5E+02*epmach,0.5E-28)) then
   begin
   ier := 6;
   exit;
   end;

{c
c
c           first approximation to the integral
c           -----------------------------------
c
c           determine the interval to be mapped onto (0,1).
c           if inf := 2 the integral is computed as i := i1+i2, where
c           i1 := integral of f over (-infinity,0),
c           i2 := integral of f over (0,+infinity).
c}
boun := bound;
if (inf=2) then boun := 0.0;
dqk15i(f,p,boun,inf,0.0,0.1E+01,result,abserr,defabs,resabs);
{c
c           test on accuracy
c}
last := 1;
rlist[1] := result;
elist[1] := abserr;
iord[1] := 1;
dres := abs(result);
errbnd := dmax1(epsabs,epsrel*dres);
if(abserr<=1.0E+02*epmach*defabs) and (abserr>errbnd) then ier := 2;
if(limit=1) then ier := 1;
if(ier<>0) or ((abserr<=errbnd) and (abserr<>resabs)) or
     (abserr=0.0) then
     begin
     neval := 30*last-15;
     if(inf=2) then neval := 2*neval;
     if(ier>2) then ier:=ier-1;
     exit;
     end;

{c
c           initialization
c           --------------
c}
uflow := d1mach(1);
oflow := d1mach(2);
rlist2[1] := result;
errmax := abserr;
maxerr := 1;
area := result;
errsum := abserr;
abserr := oflow;
nrmax := 1;
nres := 0;
ktmin := 0;
numrl2 := 2;
extrap := false;
noext := false;
ierro := 0;
iroff1 := 0;
iroff2 := 0;
iroff3 := 0;
ksgn := -1;
if(dres>=(0.1E+01-0.5E+02*epmach)*defabs) then ksgn := 1;
{c
c           main do-loop
c           ------------
c}
last:=2;
while last<=limit do
   begin
{c
c           bisect the subinterval with nrmax-th largest error estimate.
c}
   a1 := alist[maxerr];
   b1 := 0.5*(alist[maxerr]+blist[maxerr]);
   a2 := b1;
   b2 := blist[maxerr];
   erlast := errmax;
   dqk15i(f,p,boun,inf,a1,b1,area1,error1,resabs,defab1);
   dqk15i(f,p,boun,inf,a2,b2,area2,error2,resabs,defab2);
{c
c           improve previous approximations to integral
c           and error and test for accuracy.
c}
   area12 := area1+area2;
   erro12 := error1+error2;
   errsum := errsum+erro12-errmax;
   area := area+area12-rlist[maxerr];
   if(defab1<>error1) and (defab2<>error2) then
      begin
      if(abs(rlist[maxerr]-area12)<=0.1E-04*abs(area12)) and
        (erro12>=0.99E+00*errmax) then
        begin
        if(extrap) then iroff2 := iroff2+1;
        if(not extrap) then iroff1 := iroff1+1;
        end;
      if(last>10) and (erro12>errmax) then iroff3 := iroff3+1;
      end;
   rlist[maxerr] := area1;
   rlist[last] := area2;
   errbnd := dmax1(epsabs,epsrel*abs(area));
{c
c           test for roundoff error and eventually set error flag.
c}
   if(iroff1+iroff2>=10) or (iroff3>=20) then ier := 2;
   if(iroff2>=5) then ierro := 3;
{c
c           set error flag in the case that the number of
c           subintervals equals limit.
c}
   if(last=limit) then ier := 1;
{c
c           set error flag in the case of bad integrand behaviour
c           at some points of the integration range.
c}
   if(dmax1(abs(a1),abs(b2))<=(0.1E+01+0.1E+03*epmach)*
       (abs(a2)+0.1E+04*uflow)) then ier := 4;
{c
c           append the newly-created intervals to the list.
c}
   if (error2<=error1) then
        begin
        alist[last] := a2;
        blist[maxerr] := b1;
        blist[last] := b2;
        elist[maxerr] := error1;
        elist[last] := error2;
        end
   else
        begin
        alist[maxerr] := a2;
        alist[last] := a1;
        blist[last] := b1;
        rlist[maxerr] := area2;
        rlist[last] := area1;
        elist[maxerr] := error2;
        elist[last] := error1;
        end;
{c
c           call subroutine dqpsrt to maintain the descending ordering
c           in the list of error estimates and select the subinterval
c           with nrmax-th largest error estimate (to be bisected next).
c}
   dqpsrt(limit,last,maxerr,errmax,elist,iord,nrmax);
   if(errsum<=errbnd) then goto lab115;
   if(ier<>0) then goto lab100;
   if(last=2) then
      begin
      small := 0.375;
      erlarg := errsum;
      ertest := errbnd;
      rlist2[2] := area;
      end
   else if(not noext) then
      begin
      erlarg := erlarg-erlast;
      if(abs(b1-a1)>small) then erlarg := erlarg+erro12;
      if(not extrap) then
         begin
{c
c           test whether the interval to be bisected next is the
c           smallest interval.
c}
         if(abs(blist[maxerr]-alist[maxerr])>small) then goto lab90;
         extrap := true;
         nrmax := 2;
         end;
      if(ierro<>3) and (erlarg>ertest) then
        begin
{c
c           the smallest interval has the largest error.
c           before bisecting decrease the sum of the errors over the
c           larger intervals (erlarg) and perform extrapolation.
c}
        id := nrmax;
        jupbnd := last;
        if(last>(2+limit div 2)) then jupbnd := limit+3-last;
        for k := id to jupbnd do
           begin
           maxerr := iord[nrmax];
           errmax := elist[maxerr];
           if(abs(blist[maxerr]-alist[maxerr])>small) then goto lab90;
           nrmax := nrmax+1;
           end;
        end;
{c
c           perform extrapolation.
c}
      numrl2 := numrl2+1;
      rlist2[numrl2] := area;
      dqelg(numrl2,rlist2,reseps,abseps,res3la,nres);
      ktmin := ktmin+1;
      if(ktmin>5) and (abserr<0.1E-02*errsum) then ier := 5;
      if(abseps<abserr) then
           begin
           ktmin := 0;
           abserr := abseps;
           result := reseps;
           correc := erlarg;
           ertest := dmax1(epsabs,epsrel*abs(reseps));
           if(abserr<=ertest) then goto lab100;
           end;
{c
c            prepare bisection of the smallest interval.
c}
      if(numrl2=1) then noext := true;
      if(ier=5) then goto lab100;
      maxerr := iord[1];
      errmax := elist[maxerr];
      nrmax := 1;
      extrap := false;
      small := small*0.5;
      erlarg := errsum;
      end;
lab90:   last:=last+1;
   end;{90}

{c
c           set final result and error estimate.
c           ------------------------------------
c}
lab100: if(abserr=oflow) then goto lab115;
if((ier+ierro)=0) then goto lab110;
if(ierro=3) then abserr := abserr+correc;
if(ier=0) then ier := 3;
if(result<>0.0) and (area<>0.0)then goto lab105;
if(abserr>errsum) then goto lab115;
if(area=0.0) then goto lab130;
goto lab110;
lab105: if(abserr/abs(result)>errsum/abs(area))then goto lab115;
{c
c           test on divergence
c}
lab110: if(ksgn= -1) and (dmax1(abs(result),abs(area))<=defabs*0.1E-01) then
       goto lab130;
if(0.1E-01>(result/area)) or ((result/area)>0.1E+03) or
    (errsum>abs(area)) then ier := 6;
goto lab130;
{c
c           compute global integral sum.
c}
lab115: result := 0.0;
for k := 1 to last do
        result := result+rlist[k];
abserr := errsum;
lab130: neval := 30*last-15;
if(inf=2) then neval := 2*neval;
if(ier>2) then ier:=ier-1;
end;

procedure dqagse(f: integfunc; p:pointer; a,b,epsabs,epsrel: double; limit: integer;
                 var result,abserr: double; var neval,ier: integer;
                 var alist,blist,rlist,elist: nvec; var iord: nintvec;
                 var last: integer);

var
abseps,area,area1,area12,area2,a1,a2,b1,b2,correc,defabs,defab1,defab2,
dres,epmach,erlarg,erlast,errbnd,errmax,error1,error2,erro12,errsum,ertest,
oflow,resabs,reseps,small,uflow: double;
id,ierro,iroff1,iroff2,iroff3,jupbnd,k,ksgn,ktmin,maxerr,
nres,nrmax,numrl2: integer;
extrap,noext: boolean;

res3la: vec3; rlist2: vec52;

label
lab80, lab90, lab100, lab115, lab130;

begin
epmach := d1mach(4);
{c
c            test on validity of parameters
c            ------------------------------}
ier := 0;
neval := 0;
last := 0;
result := 0.0;
abserr := 0.0;
alist[1] := a;
blist[1] := b;
rlist[1] := 0.0;
elist[1] := 0.0;
if(epsabs<=0.0) and (epsrel<dmax1(0.5E+02*epmach,0.5E-28)) then
   begin
   ier := 6;
   exit;
   end;
{c
c           first approximation to the integral
c           -----------------------------------
c}
uflow := d1mach(1);
oflow := d1mach(2);
ierro := 0;
dqk21(f,p,a,b,result,abserr,defabs,resabs);
{c
c           test on accuracy.
c}
dres := abs(result);
errbnd := dmax1(epsabs,epsrel*dres);
last := 1;
rlist[1] := result;
elist[1] := abserr;
iord[1] := 1;
if(abserr<=1.0E+02*epmach*defabs) and (abserr>errbnd) then
   ier := 2;
if(limit=1) then ier := 1;
if(ier<>0) or ((abserr<=errbnd) and (abserr<>resabs)) or (abserr=0.0) then
   begin
   neval := 42*last-21;
   exit;
   end;
{c
c           initialization
c           --------------
c}
rlist2[1] := result;
errmax := abserr;
maxerr := 1;
area := result;
errsum := abserr;
abserr := oflow;
nrmax := 1;
nres := 0;
numrl2 := 2;
ktmin := 0;
extrap := false;
noext := false;
iroff1 := 0;
iroff2 := 0;
iroff3 := 0;
ksgn := -1;
small := abs(b-a)*0.375; {Adicionado em 09/05/2001}
ertest:= errbnd; {Adicionado em 09/05/2001}

if(dres>=(0.1E+01-0.5E+02*epmach)*defabs) then ksgn := 1;
{c
c           main do-loop
c           ------------
c}
last:=2;
while last<= limit do
   begin
{c
c           bisect the subinterval with the nrmax-th largest error
c           estimate.
c}
   a1 := alist[maxerr];
   b1 := 0.5*(alist[maxerr]+blist[maxerr]);
   a2 := b1;
   b2 := blist[maxerr];
   erlast := errmax;
   dqk21(f,p,a1,b1,area1,error1,resabs,defab1);
   dqk21(f,p,a2,b2,area2,error2,resabs,defab2);
{c
c           improve previous approximations to integral
c           and error and test for accuracy.
c}
   area12 := area1+area2;
   erro12 := error1+error2;
   errsum := errsum+erro12-errmax;
   area := area+area12-rlist[maxerr];
   if(defab1<>error1) and (defab2<>error2) then
      begin
      if(abs(rlist[maxerr]-area12)<=0.1E-04*abs(area12)) and
         (erro12>=0.99*errmax) then
         if(extrap) then iroff2 := iroff2+1
         else iroff1 := iroff1+1;
      if(last>10) and (erro12>errmax) then iroff3 := iroff3+1;
      end;
   rlist[maxerr] := area1;
   rlist[last] := area2;
   errbnd := dmax1(epsabs,epsrel*abs(area));
{c
c           test for roundoff error and eventually set error flag.
c}
   if(iroff1+iroff2>=10) or (iroff3>=20) then ier := 2;
   if(iroff2>=5) then ierro := 3;
{c
c           set error flag in the case that the number of subintervals
c           equals limit.
c}
   if(last=limit) then ier := 1;
{c
c           set error flag in the case of bad integrand behaviour
c           at a point of the integration range.
c}
   if dmax1(abs(a1),abs(b2))<=(1.0+0.1E+03*epmach)*(abs(a2)+0.1E+04*uflow) then
       ier := 4;
{c
c           append the newly-created intervals to the list.
c}
   if(error2<=error1) then
      begin
      alist[last] := a2;
      blist[maxerr] := b1;
      blist[last] := b2;
      elist[maxerr] := error1;
      elist[last] := error2;
      end
   else
      begin
      alist[maxerr] := a2;
      alist[last] := a1;
      blist[last] := b1;
      rlist[maxerr] := area2;
      rlist[last] := area1;
      elist[maxerr] := error2;
      elist[last] := error1;
      end;
{c
c           call subroutine dqpsrt to maintain the descending ordering
c           in the list of error estimates and select the subinterval
c           with nrmax-th largest error estimate (to be bisected next).
c}
   dqpsrt(limit,last,maxerr,errmax,elist,iord,nrmax);
{c ***jump out of do-loop}
   if(errsum<=errbnd) then goto lab115;
{c ***jump out of do-loop}
   if(ier<>0) then goto lab100;
   if(last=2) then goto lab80;
   if(noext) then goto lab90;
   erlarg := erlarg-erlast;
   if(abs(b1-a1)>small) then erlarg := erlarg+erro12;
   if(not extrap) then
      begin
{c
c           test whether the interval to be bisected next is the
c           smallest interval.
c}
      if(abs(blist[maxerr]-alist[maxerr])>small) then goto lab90;
      extrap := true;
      nrmax := 2;
      end;
   if(ierro<>3) and (erlarg>ertest) then
      begin
{c
c           the smallest interval has the largest error.
c           before bisecting decrease the sum of the errors over the
c           larger intervals (erlarg) and perform extrapolation.
c}
      id := nrmax;
      jupbnd := last;
      if(last>(2+limit div 2)) then jupbnd := limit+3-last;
      for k := id to jupbnd do
         begin
         maxerr := iord[nrmax];
         errmax := elist[maxerr];
{c ***jump out of do-loop}
         if(abs(blist[maxerr]-alist[maxerr])>small) then goto lab90;
         nrmax := nrmax+1;
         end
      end;
{c
c           perform extrapolation.
c}
   numrl2 := numrl2+1;
   rlist2[numrl2] := area;
   dqelg(numrl2,rlist2,reseps,abseps,res3la,nres);
   ktmin := ktmin+1;
   if(ktmin>5) and (abserr<0.1E-02*errsum) then ier := 5;
   if(abseps<abserr) then
      begin
      ktmin := 0;
      abserr := abseps;
      result := reseps;
      correc := erlarg;
      ertest := dmax1(epsabs,epsrel*abs(reseps));
{c ***jump out of do-loop}
      if(abserr<=ertest) then goto lab100;
      end;
{c
c           prepare bisection of the smallest interval.
c}
   if(numrl2=1) then noext := true;
   if(ier=5) then goto lab100;
   maxerr := iord[1];
   errmax := elist[maxerr];
   nrmax := 1;
   extrap := false;
   small := small*0.5;
   erlarg := errsum;
   goto lab90;
   lab80:   small := abs(b-a)*0.375;
   erlarg := errsum;
   ertest := errbnd;
   rlist2[2] := area;
   lab90: last:= last+1;
   end;
{c
c           set final result and error estimate.
c           ------------------------------------
c}
lab100: if(abserr=oflow) then goto lab115;
if(ier+ierro<>0) then
   begin
   if(ierro=3) then abserr := abserr+correc;
   if(ier=0) then ier := 3;
   if(result<>0.0) and (area<>0.0) then
      if(abserr/abs(result))>(errsum/abs(area)) then goto lab115
      else
         begin
         if(abserr>errsum) then goto lab115;
         if(area=0.0) then goto lab130;
         end;
   end;
{c
c           test on divergence.
c}
if(ksgn=(-1)) and (dmax1(abs(result),abs(area))<=defabs*0.1E-01) then
    goto lab130;
if(0.1E-01>(result/area)) or ((result/area)>0.1E+03) or
   (errsum>abs(area)) then ier := 6;
goto lab130;
{c
c           compute global integral sum.
c}
lab115:  result := 0.0;
for k := 1 to last do
         result := result+rlist[k];
abserr := errsum;
lab130: if(ier>2) then ier := ier-1;
neval := 42*last-21;
end;


procedure dqawoe (f: integfunc; p:pointer; a,b,omega: double; integr: integer;
                 epsabs,epsrel: double; limit,icall,maxp1: integer;
                 var result,abserr: double; var neval,ier,last:integer;
                 var alist,blist,rlist,elist: nvec; var iord, nnlog: nintvec;
                 var momcom: integer; var chebmo: n25array);

var
abseps,area,area1,area12,area2,a1,
a2,b1,b2,correc,defab1,defab2,defabs,
domega,dres,epmach,erlarg,erlast,
errbnd,errmax,error1,erro12,error2,errsum,ertest,oflow,
resabs,reseps,small,uflow,width: double;

id,ierro,iroff1,iroff2,iroff3,
jupbnd,k,ksgn,ktmin,maxerr,nev,
nres,nrmax,nrmom,numrl2: integer;

extrap,noext,extall: boolean;

rlist2: vec52;
res3la: vec3;

label
lab70, lab120, lab140, lab150, lab170, lab165, lab190;

begin
epmach := d1mach(4);
{c
c         test on validity of parameters
c         ------------------------------
c}
ier := 0;
neval := 0;
last := 0;
result := 0.0;
abserr := 0.0;
alist[1] := a;
blist[1] := b;
rlist[1] := 0.0;
elist[1] := 0.0;
iord[1] := 0;
nnlog[1] := 0;
if((integr<>1) and (integr<>2)) or ((epsabs<=0.0) and
    (epsrel<dmax1(0.5E+02*epmach,0.5E-28))) or (icall<1) or (maxp1<1) then
       begin
       ier := 6;
       exit;
       end;

{c
c           first approximation to the integral
c           -----------------------------------
c}
domega := abs(omega);
nrmom := 0;
if (icall<=1) then
      momcom := 0;
dqc25f(f,p,a,b,domega,integr,nrmom,maxp1,0,result,abserr,
       neval,defabs,resabs,momcom,chebmo);
{c
c           test on accuracy.
c}
dres := abs(result);
errbnd := dmax1(epsabs,epsrel*dres);
rlist[1] := result;
elist[1] := abserr;
iord[1] := 1;
if(abserr<=0.1E+03*epmach*defabs) and (abserr>errbnd) then ier := 2;
if(limit=1) then ier := 1;
if(ier<>0) or (abserr<=errbnd) then
   begin
   if (integr=2) and (omega<0.0) then result:=-result;
   exit;
   end;


{c
c           initializations
c           ---------------
c}
uflow := d1mach(1);
oflow := d1mach(2);
errmax := abserr;
maxerr := 1;
area := result;
errsum := abserr;
abserr := oflow;
nrmax := 1;
extrap := false;
noext := false;
ierro := 0;
iroff1 := 0;
iroff2 := 0;
iroff3 := 0;
ktmin := 0;
small := abs(b-a)*0.75;
nres := 0;
numrl2 := 0;
extall := false;
if(0.5*abs(b-a)*domega<=0.2E+01) then
      begin
      numrl2 := 1;
      extall := true;
      rlist2[1] := result;
      end;
if(0.25*abs(b-a)*domega<=0.2E+01) then extall := true;
ksgn := -1;
if(dres>=(0.1E+01-0.5E+02*epmach)*defabs) then ksgn := 1;
{c
c           main do-loop
c           ------------
c}
last:=2;
while last<= limit do
   begin
{c
c           bisect the subinterval with the nrmax-th largest
c           error estimate.
c}
   nrmom := nnlog[maxerr]+1;
   a1 := alist[maxerr];
   b1 := 0.5*(alist[maxerr]+blist[maxerr]);
   a2 := b1;
   b2 := blist[maxerr];
   erlast := errmax;
   dqc25f(f,p,a1,b1,domega,integr,nrmom,maxp1,0,
       area1,error1,nev,resabs,defab1,momcom,chebmo);
   neval := neval+nev;
   dqc25f(f,p,a2,b2,domega,integr,nrmom,maxp1,1,
       area2,error2,nev,resabs,defab2,momcom,chebmo);
   neval := neval+nev;
{c
c           improve previous approximations to integral
c           and error and test for accuracy.
c}
   area12 := area1+area2;
   erro12 := error1+error2;
   errsum := errsum+erro12-errmax;
   area := area+area12-rlist[maxerr];
   if(defab1<>error1) and (defab2<>error2) then
      begin
      if(abs(rlist[maxerr]-area12)<=0.1E-04*abs(area12))
         and (erro12>=0.99*errmax) then
           begin
           if(extrap) then iroff2 := iroff2+1
           else iroff1 := iroff1+1;
           end;
      if(last>10) and (erro12>errmax) then iroff3 := iroff3+1;
      end;
   rlist[maxerr] := area1;
   rlist[last] := area2;
   nnlog[maxerr] := nrmom;
   nnlog[last] := nrmom;
   errbnd := dmax1(epsabs,epsrel*abs(area));
{c
c           test for roundoff error and eventually set error flag.
c}
   if(iroff1+iroff2>=10) or (iroff3>=20) then ier := 2;
   if(iroff2>=5) then ierro := 3;
{c
c           set error flag in the case that the number of
c           subintervals equals limit.
c}
   if(last=limit) then ier := 1;
{c
c           set error flag in the case of bad integrand behaviour
c           at a point of the integration range.
c}
   if(dmax1(abs(a1),abs(b2))<=(0.1E+01+0.1E+03*epmach)
       *(abs(a2)+0.1E+04*uflow)) then ier := 4;
{c
c           append the newly-created intervals to the list.
c}
   if(error2<=error1) then
        begin
        alist[last] := a2;
        blist[maxerr] := b1;
        blist[last] := b2;
        elist[maxerr] := error1;
        elist[last] := error2;
        end
   else
        begin
        alist[maxerr] := a2;
        alist[last] := a1;
        blist[last] := b1;
        rlist[maxerr] := area2;
        rlist[last] := area1;
        elist[maxerr] := error2;
        elist[last] := error1;
        end;
{c
c           call subroutine dqpsrt to maintain the descending ordering
c           in the list of error estimates and select the subinterval
c           with nrmax-th largest error estimate (to bisected next).
c}
   dqpsrt(limit,last,maxerr,errmax,elist,iord,nrmax);
{c ***jump out of do-loop}
   if(errsum<=errbnd) then goto lab170;
   if(ier<>0) then goto lab150;
   if(last=2) and (extall) then goto lab120;
   if(noext) then goto lab140;

   if(extall) then
        begin
        erlarg := erlarg-erlast;
        if(abs(b1-a1)>small) then erlarg := erlarg+erro12;
        if(extrap) then goto lab70;
        end;
{c
c           test whether the interval to be bisected next is the
c           smallest interval.
c}
   width := abs(blist[maxerr]-alist[maxerr]);
   if(width>small) then goto lab140;
   if(not extall) then
      begin
{c
c           test whether we can start with the extrapolation procedure
c           (we do this if we integrate over the next interval with
c           use of a gauss-kronrod rule - see subroutine dqc25f).
c}
      small := small*0.5;
      if(0.25*width*domega>0.2E+01) then goto lab140;
      extall := true;
      end
   else
      begin
      extrap := true;
      nrmax := 2;
      lab70: if(ierro<>3) and (erlarg>ertest) then
         begin
{c
c           the smallest interval has the largest error.
c           before bisecting decrease the sum of the errors over
c           the larger intervals (erlarg) and perform extrapolation.
c}
         jupbnd := last;
         if (last>(limit div 2+2)) then jupbnd := limit+3-last;
         id := nrmax;
         for  k := id to jupbnd do
            begin
            maxerr := iord[nrmax];
            errmax := elist[maxerr];
            if(abs(blist[maxerr]-alist[maxerr])>small) then goto lab140;
            nrmax := nrmax+1;
            end;
         end;
{c
c           perform extrapolation.
c}
      numrl2 := numrl2+1;
      rlist2[numrl2] := area;
      if(numrl2>=3) then
         begin
         dqelg(numrl2,rlist2,reseps,abseps,res3la,nres);
         ktmin := ktmin+1;
         if(ktmin>5) and (abserr<0.1E-02*errsum) then ier := 5;
         if(abseps<abserr) then
            begin
            ktmin := 0;
            abserr := abseps;
            result := reseps;
            correc := erlarg;
            ertest := dmax1(epsabs,epsrel*abs(reseps));
            { ***jump out of do-loop}
            if(abserr<=ertest) then goto lab150
            end;
   {c
   c           prepare bisection of the smallest interval.
   c}
         if(numrl2=1) then noext := true;
         if(ier=5) then goto lab150;
         end;
      maxerr := iord[1];
      errmax := elist[maxerr];
      nrmax := 1;
      extrap := false;
      small := small*0.5;
      erlarg := errsum;
      goto lab140;
      lab120: small := small*0.5;
      numrl2 := numrl2+1;
      rlist2[numrl2] := area;
      end;

   ertest := errbnd;
   erlarg := errsum;
   lab140: last:=last+1;
   end;
{c
c           set the final result.
c           ---------------------
c}
lab150: if(abserr=oflow) or (nres=0) then goto lab170;
if(ier+ierro=0) then goto lab165;
if(ierro=3) then abserr := abserr+correc;
if(ier=0) then ier := 3;
if(result=0.0) or (area=0.0) then
   begin
   if(abserr>errsum) then goto lab170;
   if(area=0.0) then goto lab190;
   end
else
   if(abserr/abs(result)>errsum/abs(area)) then goto lab170;
{c
c           test on divergence.
c}
lab165: if(ksgn=(-1)) and (dmax1(abs(result),abs(area))<=defabs*0.1E-01) then
    goto lab190;
if(0.1E-01 > (result/area)) or ((result/area)>0.1E+03) or
  (errsum>=abs(area)) then
  ier := 6;
goto lab190;
{c
c           compute global integral sum.
c}
lab170: result := 0.0;
for k:=1 to last do
        result := result+rlist[k];
abserr := errsum;
lab190: if (ier>2) then ier:=ier-1;
if (integr=2) and (omega<0.0) then result:=-result;
end;

procedure dqawfe(f:integfunc; p:pointer; a,omega: double; integr: integer;
                 epsabs: double; limlst,limit,maxp1: integer;
                 var result,abserr: double; var neval,ier: integer;
                 var rslst,erlst: nvec; var ierlst: nintvec;
                 var lst: integer; var alist,blist,rlist,elist: nvec;
                 var iord,nnlog: nintvec; var chebmo: n25array);

const
pconst= 0.9;

var
abseps,correc,cycle,c1,c2,dabs,dl,dla,drl,ep,eps,epsa,
errsum,fact,p1,reseps,uflow: double;

ktmin,last,ll,momcom,nev,nres,numrl2: integer;
l: longint;

psum: vec52;
res3la: vec3;

label
lab60, lab80;

begin
result := 0.0E+00;
abserr := 0.0E+00;
neval := 0;
lst := 0;
ier := 0;
if((integr<>1) and (integr<>2)) or (epsabs<0.0E+00) or (limlst<3) then
   ier := 6;
if(ier=6) then exit;

if(omega=0.0E+00) then
   begin
{c
c           integration by dqagie if omega is zero
c           --------------------------------------
c}
   if(integr=1) then
      dqagie(f,p,{0.0E+00}a,1,epsabs,0.0E+00,limit,
             result,abserr,neval,ier,alist,blist,rlist,elist,iord,last);
   rslst[1] := result;
   erlst[1] := abserr;
   ierlst[1] := ier;
   lst := 1;
   exit;
   end;
{c
c           initializations
c           ---------------
c}
l := trunc(abs(omega));
dl := 2*l+1;
cycle := dl*pi/abs(omega);
ier := 0;
ktmin := 0;
neval := 0;
numrl2 := 0;
nres := 0;
c1 := a;
c2 := cycle+a;
p1 := 0.1E+01-pconst;
uflow := d1mach(1);
eps := epsabs;
if(epsabs>uflow/p1) then eps := epsabs*p1;
ep := eps;
fact := 0.1E+01;
correc := 0.0E+00;
abserr := 0.0E+00;
errsum := 0.0E+00;
{c
c           main do-loop
c           ------------
c}
lst:= 1;
while lst<=limlst do
      {do 50 lst := 1,limlst}
   begin
{c
c           integrate over current subinterval.
c}
   dla := lst;
   epsa := eps*fact;
   dqawoe(f,p,c1,c2,omega,integr,epsa,0.0E+00,limit,lst,maxp1,
          rslst[lst],erlst[lst],nev,ierlst[lst],last,alist,blist,rlist,
           elist,iord,nnlog,momcom,chebmo);
   neval := neval+nev;
   fact := fact*pconst;
   errsum := errsum+erlst[lst];
   drl := 0.5E+02*abs(rslst[lst]);
{c
c           test on accuracy with partial sum
c}
   if((errsum+drl)<=epsabs) and (lst>=6) then goto lab80;
   correc := dmax1(correc,erlst[lst]);
   if(ierlst[lst]<>0) then eps := dmax1(ep,correc*p1);
   if(ierlst[lst]<>0) then ier := 7;
   if(ier=7) and ((errsum+drl)<=correc*0.1E+02) and (lst>5) then goto lab80;
   numrl2 := numrl2+1;
   if(lst<=1) then
        psum[1] := rslst[1]
   else
      begin
      psum[numrl2] := psum[ll]+rslst[lst];
      if(lst<>2) then
         begin
{c
c           test on maximum number of subintervals
c}
         if(lst=limlst) then ier := 1;
{c
c           perform new extrapolation
c}
         dqelg(numrl2,psum,reseps,abseps,res3la,nres);
{c
c           test whether extrapolated result is influenced by roundoff
c}
         ktmin := ktmin+1;
         if(ktmin>=15) and (abserr<=0.1E-02*(errsum+drl)) then ier := 4;
         if(abseps<=abserr) or (lst=3) then
            begin
            abserr := abseps;
            result := reseps;
            ktmin := 0;
{c
c           if ier is not 0, check whether direct result (partial sum)
c           or extrapolated result yields the best integral
c           approximation
c}
            if((abserr+0.1E+02*correc)<=epsabs) or
              ((abserr<=epsabs) and (0.1E+02*correc>=epsabs)) then goto lab60;
            end;
         if(ier<>0) and (ier<>7) then goto lab60;
         end;
      end;
   ll := numrl2;
   c1 := c2;
   c2 := c2+cycle;
   lst:= lst+1;
   end;
{c
c         set final result and error estimate
c         -----------------------------------
c}
lab60: abserr := abserr+0.1E+02*correc;
if(ier=0) then  exit;
if(result=0.0E+00) or (psum[numrl2]=0.0E+00) then
   begin
   if(abserr>errsum) then goto lab80;
   if(psum[numrl2]=0.0E+00) then exit;
   end;
if(abserr/abs(result)<=(errsum+drl)/abs(psum[numrl2])) then
   if(ier>=1) and (ier<>7) then abserr := abserr+drl
else
   begin
   lab80: result := psum[numrl2];
   abserr := errsum+drl;
   end;
end;

procedure dqawf(f: integfunc; p:pointer;a,omega: double; integr: integer;
                epsabs: double; var result,abserr: double;
                var neval,ier: integer; limlst:integer;
                var lst:integer; leniw,maxp1,lenw: integer;
                var iwork: nintvec ; var work: nvec);

var
last,limit,ll2,lvl,l1,l2,l3,l4,l5,l6: integer;

work1, workl1, workl2, workl3, workl4, workl5: ^nvec;
iwork1, iworkl1, iworkll2: ^nintvec;
workl6: ^n25array;

begin
ier := 6;
neval := 0;
last := 0;
result := 0.0E+00;
abserr := 0.0E+00;
if(limlst>=3) and (leniw>=(limlst+2)) and (maxp1>=1) and
    (lenw>=(leniw*2+maxp1*25)) then
   begin
{c
c         prepare call for dqawfe
c}
   limit := (leniw-limlst) div 2;
   l1 := limlst+1;
   l2 := limlst+l1;
   l3 := limit+l2;
   l4 := limit+l3;
   l5 := limit+l4;
   l6 := limit+l5;
   ll2 := limit+l1;
   work1:=@work[1];
   workl1:=@work[l1];
   iwork1:=@iwork[1];
   workl2:=@work[l2];
   workl3:=@work[l3];
   workl4:=@work[l4];
   workl5:=@work[l5];
   iworkl1:=@iwork[l1];
   iworkll2:=@iwork[ll2];
   workl6:=@work[l6];

   dqawfe(f,p,a,omega,integr,epsabs,limlst,limit,maxp1,result,
       abserr,neval,ier,work1^,workl1^,iwork1^,lst,workl2^,
       workl3^,workl4^,workl5^,iworkl1^,iworkll2^,workl6^);
{c
c         call error handler if necessary
c}
   lvl := 0;
   end;
if(ier=6) then lvl := 1;
{if(ier<>0) and (ier<>5) then
   writeln('Abnormal return from dqawf. Error: ',ier,'  Level: ',lvl);}
end;

procedure dqawo(f: integfunc; p:pointer;a,b,omega: double; integr: integer;
                epsabs,epsrel: double; var result,abserr: double;
                var neval,ier:integer; leniw,maxp1,lenw: integer;
                var last: integer; var iwork: nintvec; var work: nvec);

var
limit,lvl,l1,l2,l3,l4,momcom: integer;
work1,workl1,workl2,workl3: ^nvec;
iwork1, iworkl1: ^nintvec;
workl4: ^n25array;


begin
ier := 6;
neval := 0;
last := 0;
result := 0.0;
abserr := 0.0;
if(leniw>=2) and (maxp1>=1) and (lenw>=(leniw*2+maxp1*25)) then
   begin
{c
c         prepare call for dqawoe
c}
   limit := leniw div 2;
   l1 := limit+1;
   l2 := limit+l1;
   l3 := limit+l2;
   l4 := limit+l3;
   work1:=@work[1];
   workl1:=@work[l1];
   workl2:=@work[l2];
   workl3:=@work[l3];
   iwork1:=@iwork[1];
   iworkl1:=@iwork[l1];
   workl4:=@work[l4];
   dqawoe(f,p,a,b,omega,integr,epsabs,epsrel,limit,1,maxp1,result,
          abserr,neval,ier,last,work1^,workl1^,workl2^,workl3^,
          iwork1^,iworkl1^,momcom,workl4^);
{c
c         call error handler if necessary
c}
   lvl := 0;
   end;
if(ier=6) then lvl := 0;
{if(ier<>0) and (ier<>5) then
   writeln('abnormal return from dqawo. Error= ',ier,'  Level= ',lvl);}
end;

procedure dqags(f: integfunc; p:pointer; a,b,epsabs,epsrel: double;
                var result,abserr: double; var neval,ier: integer;
                limit,lenw: integer; var last: integer;
                var iwork: nintvec; var work: nvec);

var
lvl,l1,l2,l3: integer;
workl1, workl2, workl3: ^nvec;

begin
ier := 6;
neval := 0;
last := 0;
result := 0.0;
abserr := 0.0;
if(limit>=1) and (lenw>=limit*4) then
   begin
{c
c         prepare call for dqagse.
c}
   l1 := limit+1;
   l2 := limit+l1;
   l3 := limit+l2;
   workl1:=@work[l1];
   workl2:=@work[l2];
   workl3:=@work[l3];

   dqagse(f,p,a,b,epsabs,epsrel,limit,result,abserr,neval,
          ier,work,workl1^,workl2^,workl3^,iwork,last);
{c
c         call error handler if necessary.
c}
   lvl := 0;
   end;
if(ier=6) then lvl := 1;
{if(ier<>0) and (ier<>5) then
   writeln('abnormal return from dqags. Error= ',ier,'  Level= ',lvl,
   '  a=',a:10:5, '  b=', b:10:5);}
end;



procedure dqagi(f: integfunc; p:pointer; bound: double; inf: integer;
                epsabs,epsrel: double; var result,abserr: double;
                var neval,ier: integer; limit,lenw: integer;
                var last: integer; var iwork: nintvec; var work: nvec);
var
lvl,l1,l2,l3: integer;
l1ptr, l2ptr, l3ptr: ^nvec;
begin
ier := 6;
neval := 0;
last := 0;
result := 0.0;
abserr := 0.0;
if (limit>=1) and (lenw>=limit*4) then
   begin
{c
c         prepare call for dqagie.
c}
   l1 := limit+1; l1ptr:=@work[l1];
   l2 := limit+l1; l2ptr:=@work[l2];
   l3 := limit+l2; l3ptr:=@work[l3];

   dqagie(f,p,bound,inf,epsabs,epsrel,limit,result,abserr,
          neval,ier,work,l1ptr^,l2ptr^,l3ptr^,iwork,last);
{c
c         call error handler if necessary.
c}
   lvl := 0;
   end;
if(ier=6) then lvl := 1;
{if(ier<>0) and (ier<>5) then
   writeln('Abnormal return from dqagi. Error= ',ier,'   Level= ',lvl);}
end;

function dqwgtc(x, c, p2, p3, p4: double; kp: integer): double;
begin
dqwgtc:= 1.0/(x-c);
end;

procedure dqc25c(f: integfunc; p: pointer; a,b,c: double; 
                 var result,abserr: double; var krul,neval: integer);
{c***begin prologue  dqc25c
c***date written   810101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a2a2,j4
c***keywords  25-point clenshaw-curtis integration
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math. & progr. div. - k.u.leuven
c***purpose  to compute i = integral of f*w over (a,b) with
c            error estimate, where w(x) = 1/(x-c)
c***description
c
c        integration rules for the computation of cauchy
c        principal value integrals
c        standard fortran subroutine
c        double precision version
c
c        parameters
c           f      - double precision
c                    function subprogram defining the integrand function
c                    f(x). the actual name for f needs to be declared
c                    e x t e r n a l  in the driver program.
c
c           a      - double precision
c                    left end point of the integration interval
c
c           b      - double precision
c                    right end point of the integration interval, b.gt.a
c
c           c      - double precision
c                    parameter in the weight function
c
c           result - double precision
c                    approximation to the integral
c                    result is computed by using a generalized
c                    clenshaw-curtis method if c lies within ten percent
c                    of the integration interval. in the other case the
c                    15-point kronrod rule obtained by optimal addition
c                    of abscissae to the 7-point gauss rule, is applied.
c
c           abserr - double precision
c                    estimate of the modulus of the absolute error,
c                    which should equal or exceed abs(i-result)
c
c           krul   - integer
c                    key which is decreased by 1 if the 15-point
c                    gauss-kronrod scheme has been used
c
c           neval  - integer
c                    number of integrand evaluations
c
c.......................................................................
c***references  (none)
c***routines called  dqcheb,dqk15w,dqwgtc
c***end prologue  dqc25c
c}

var
{      double precision a,abserr,ak22,amom0,amom1,amom2,b,c,cc,centr,
     *  cheb12,cheb24,dabs,dlog,dqwgtc,f,fval,hlgth,p2,p3,p4,resabs,
     *  resasc,result,res12,res24,u,x
      integer i,isym,k,kp,krul,neval
c}
ak22, amom0, amom1, amom2, cc, centr, hlgth, p2, p3,
p4, resabs, resasc, res12, res24, u: double;
i, isym, k, kp: integer;

{      dimension x(11),fval(25),cheb12(13),cheb24(25)
c}

fval: vec25;
cheb12: vec13;
cheb24: vec25;

{      external f,dqwgtc
c
c           the vector x contains the values cos(k*pi/24),
c           k = 1, ..., 11, to be used for the chebyshev series
c           expansion of f
c}

const
x: vec11= (
0.991444861373810411144557526928563E0,
0.965925826289068286749743199728897E0,
0.923879532511286756128183189396788E0,
0.866025403784438646763723170752936E0,
0.793353340291235164579776961501299E0,
0.707106781186547524400844362104849E0,
0.608761429008720639416097542898164E0,
0.500000000000000000000000000000000E0,
0.382683432365089771728459984030399E0,
0.258819045102520762348898837624048E0,
0.130526192220051591548406227895489E0);

{c
c           list of major variables
c           ----------------------
c           fval   - value of the function f at the points
c                    cos(k*pi/24),  k = 0, ..., 24
c           cheb12 - chebyshev series expansion coefficients,
c                    for the function f, of degree 12
c           cheb24 - chebyshev series expansion coefficients,
c                    for the function f, of degree 24
c           res12  - approximation to the integral corresponding
c                    to the use of cheb12
c           res24  - approximation to the integral corresponding
c                    to the use of cheb24
c           dqwgtc - external function subprogram defining
c                    the weight function
c           hlgth  - half-length of the interval
c           centr  - mid point of the interval
c
c
c           check the position of c.
c
c***first executable statement  dqc25c}

begin
{      cc = (0.2d+01*c-b-a)/(b-a)
      if(dabs(cc).lt.0.11d+01) go to 10
c
c           apply the 15-point gauss-kronrod scheme.
c
      krul = krul-1
      call dqk15w(f,dqwgtc,c,p2,p3,p4,kp,a,b,result,abserr,
     *  resabs,resasc)
      neval = 15
      if (resasc.eq.abserr) krul = krul+1
      go to 50}
      
cc:= (2.0*c - b - a)/(b-a);
if abs(cc)>= 1.1 then
   begin
   krul:= krul-1;
   dqk15w(f, p, @dqwgtc, c, p2, p3, p4, kp, a, b, result, abserr,
          resabs, resasc);
   neval := 15;
   if resasc=abserr then krul:= krul+1;
   end
else
{c
c           use the generalized clenshaw-curtis method.
c
   10 hlgth = 0.5d+00*(b-a)
      centr = 0.5d+00*(b+a)
      neval = 25
      fval(1) = 0.5d+00*f(hlgth+centr)
      fval(13) = f(centr)
      fval(25) = 0.5d+00*f(centr-hlgth)
      do 20 i=2,12
        u = hlgth*x(i-1)
        isym = 26-i
        fval(i) = f(u+centr)
        fval(isym) = f(centr-u)
   20 continue}
   begin
   hlgth:= 0.5*(b-a);
   centr:= 0.5*(b+a);
   neval:= 25;
   fval[1]:= 0.5*f(p,hlgth+centr);
   fval[13]:= f(p,centr);
   fval[25]:= 0.5*f(p,centr-hlgth);
   for i:= 2 to 12 do
      begin
      u:= hlgth*x[i-1];
      isym:= 26-i;
      fval[i]:= f(p,u+centr);
      fval[isym]:= f(p,centr-u);
      end;
{c
c           compute the chebyshev series expansion.
c
      call dqcheb(x,fval,cheb12,cheb24)}
   dqcheb(x, fval, cheb12, cheb24);
{c
c           the modified chebyshev moments are computed by forward
c           recursion, using amom0 and amom1 as starting values.
c
      amom0 = dlog(dabs((0.1d+01-cc)/(0.1d+01+cc)))
      amom1 = 0.2d+01+cc*amom0
      res12 = cheb12(1)*amom0+cheb12(2)*amom1
      res24 = cheb24(1)*amom0+cheb24(2)*amom1
      do 30 k=3,13
        amom2 = 0.2d+01*cc*amom1-amom0
        ak22 = (k-2)*(k-2)
        if((k/2)*2.eq.k) amom2 = amom2-0.4d+01/(ak22-0.1d+01)
        res12 = res12+cheb12(k)*amom2
        res24 = res24+cheb24(k)*amom2
        amom0 = amom1
        amom1 = amom2
   30 continue}
   amom0:= ln(abs(1.0-cc)/(1.0+cc));
   amom1:= 2.0 + cc*amom0;
   res12:= cheb12[1]*amom0 + cheb12[2]*amom1;
   res24:= cheb24[1]*amom0 + cheb24[2]*amom1;
   for k:= 3 to 13 do
      begin
      amom2:= 2.0*cc*amom1-amom0;
      ak22:= (k-2)*(k-2);
      if k mod 2 = 0 then amom2:= amom2-4.0/(ak22-1.0);
      res12:= res12 + cheb12[k]*amom2;
      res24:= res24 + cheb24[k]*amom2;
      amom0:= amom1;
      amom1:= amom2;
      end;
      
{      do 40 k=14,25
        amom2 = 0.2d+01*cc*amom1-amom0
        ak22 = (k-2)*(k-2)
        if((k/2)*2.eq.k) amom2 = amom2-0.4d+01/(ak22-0.1d+01)
        res24 = res24+cheb24(k)*amom2
        amom0 = amom1
        amom1 = amom2
   40 continue}
   for k:= 14 to 25 do
      begin
      amom2:= 2.0*cc*amom1 - amom0;
      ak22:= (k-2)*(k-2);
      if k mod 2 = 0 then amom2:= amom2-4.0/(ak22-1.0);
      res24:= res24 + cheb24[k]*amom2;
      amom0:= amom1;
      amom1:= amom2;
      end;
      
{      result = res24
      abserr = dabs(res24-res12)
   50 return
      end}
   result:= res24;
   abserr:= abs(res24-res12);
   end;
end;


procedure dqawce(f:integfunc; p:pointer; a,b,c,epsabs,epsrel: double;
                 limit: integer; var result,abserr: double;
		 var neval,ier: integer; var alist,blist,rlist,elist: nvec;
		 var iord: nintvec; var last: integer);

{c***begin prologue  dqawce
c***date written   800101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a2a1,j4
c***keywords  automatic integrator, special-purpose,
c             cauchy principal value, clenshaw-curtis method
c***author  piessens,robert,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math. & progr. div. - k.u.leuven
c***  purpose  the routine calculates an approximation result to a
c              cauchy principal value i = integral of f*w over (a,b)
c              (w(x) = 1/(x-c), (c.ne.a, c.ne.b), hopefully satisfying
c              following claim for accuracy
c              abs(i-result).le.max(epsabs,epsrel*abs(i))
c***description
c
c        computation of a cauchy principal value
c        standard fortran subroutine
c        double precision version
c
c        parameters
c         on entry
c            f      - double precision
c                     function subprogram defining the integrand
c                     function f(x). the actual name for f needs to be
c                     declared e x t e r n a l in the driver program.
c
c            a      - double precision
c                     lower limit of integration
c
c            b      - double precision
c                     upper limit of integration
c
c            c      - double precision
c                     parameter in the weight function, c.ne.a, c.ne.b
c                     if c = a or c = b, the routine will end with
c                     ier = 6.
c
c            epsabs - double precision
c                     absolute accuracy requested
c            epsrel - double precision
c                     relative accuracy requested
c                     if  epsabs.le.0
c                     and epsrel.lt.max(50*rel.mach.acc.,0.5d-28),
c                     the routine will end with ier = 6.
c
c            limit  - integer
c                     gives an upper bound on the number of subintervals
c                     in the partition of (a,b), limit.ge.1
c
c         on return
c            result - double precision
c                     approximation to the integral
c
c            abserr - double precision
c                     estimate of the modulus of the absolute error,
c                     which should equal or exceed abs(i-result)
c
c            neval  - integer
c                     number of integrand evaluations
c
c            ier    - integer
c                     ier = 0 normal and reliable termination of the
c                             routine. it is assumed that the requested
c                             accuracy has been achieved.
c                     ier.gt.0 abnormal termination of the routine
c                             the estimates for integral and error are
c                             less reliable. it is assumed that the
c                             requested accuracy has not been achieved.
c            error messages
c                     ier = 1 maximum number of subdivisions allowed
c                             has been achieved. one can allow more sub-
c                             divisions by increasing the value of
c                             limit. however, if this yields no
c                             improvement it is advised to analyze the
c                             the integrand, in order to determine the
c                             the integration difficulties. if the
c                             position of a local difficulty can be
c                             determined (e.g. singularity,
c                             discontinuity within the interval) one
c                             will probably gain from splitting up the
c                             interval at this point and calling
c                             appropriate integrators on the subranges.
c                         = 2 the occurrence of roundoff error is detec-
c                             ted, which prevents the requested
c                             tolerance from being achieved.
c                         = 3 extremely bad integrand behaviour
c                             occurs at some interior points of
c                             the integration interval.
c                         = 6 the input is invalid, because
c                             c = a or c = b or
c                             (epsabs.le.0 and
c                              epsrel.lt.max(50*rel.mach.acc.,0.5d-28))
c                             or limit.lt.1.
c                             result, abserr, neval, rlist(1), elist(1),
c                             iord(1) and last are set to zero. alist(1)
c                             and blist(1) are set to a and b
c                             respectively.
c
c            alist   - double precision
c                      vector of dimension at least limit, the first
c                       last  elements of which are the left
c                      end points of the subintervals in the partition
c                      of the given integration range (a,b)
c
c            blist   - double precision
c                      vector of dimension at least limit, the first
c                       last  elements of which are the right
c                      end points of the subintervals in the partition
c                      of the given integration range (a,b)
c
c            rlist   - double precision
c                      vector of dimension at least limit, the first
c                       last  elements of which are the integral
c                      approximations on the subintervals
c
c            elist   - double precision
c                      vector of dimension limit, the first  last
c                      elements of which are the moduli of the absolute
c                      error estimates on the subintervals
c
c            iord    - integer
c                      vector of dimension at least limit, the first k
c                      elements of which are pointers to the error
c                      estimates over the subintervals, so that
c                      elist(iord(1)), ..., elist(iord(k)) with k = last
c                      if last.le.(limit/2+2), and k = limit+1-last
c                      otherwise, form a decreasing sequence
c
c            last    - integer
c                      number of subintervals actually produced in
c                      the subdivision process
c
c***references  (none)
c***routines called  d1mach,dqc25c,dqpsrt
c***end prologue  dqawce
c}
var
{      double precision a,aa,abserr,alist,area,area1,area12,area2,a1,a2,
     *  b,bb,blist,b1,b2,c,dabs,dmax1,d1mach,elist,epmach,epsabs,epsrel,
     *  errbnd,errmax,error1,erro12,error2,errsum,f,result,rlist,uflow
      integer ier,iord,iroff1,iroff2,k,krule,last,limit,maxerr,nev,
     *  neval,nrmax
c}
aa, area, area1, area12, area2, a1, a2, bb, b1, b2, epmach, errbnd,
errmax, error1, erro12, error2, errsum, uflow: double;
iroff1, iroff2, k, krule, maxerr, nev, nrmax: integer;

{      dimension alist(limit),blist(limit),rlist(limit),elist(limit),
     *  iord(limit)
c
      external f
c}
{c            list of major variables
c            -----------------------
c
c           alist     - list of left end points of all subintervals
c                       considered up to now
c           blist     - list of right end points of all subintervals
c                       considered up to now
c           rlist(i)  - approximation to the integral over
c                       (alist(i),blist(i))
c           elist(i)  - error estimate applying to rlist(i)
c           maxerr    - pointer to the interval with largest
c                       error estimate
c           errmax    - elist(maxerr)
c           area      - sum of the integrals over the subintervals
c           errsum    - sum of the errors over the subintervals
c           errbnd    - requested accuracy max(epsabs,epsrel*
c                       abs(result))
c           *****1    - variable for the left subinterval
c           *****2    - variable for the right subinterval
c           last      - index for subdivision
c
c
c            machine dependent constants
c            ---------------------------
c
c           epmach is the largest relative spacing.
c           uflow is the smallest positive magnitude.
c
c***first executable statement  dqawce}
begin
{      epmach = d1mach(4)
      uflow = d1mach(1)
c}
epmach:= d1mach(4);
uflow:= d1mach(1);

{c
c           test on validity of parameters
c           ------------------------------
c
      ier = 6
      neval = 0
      last = 0
      alist(1) = a
      blist(1) = b
      rlist(1) = 0.0d+00
      elist(1) = 0.0d+00
      iord(1) = 0
      result = 0.0d+00
      abserr = 0.0d+00
      if(c.eq.a.or.c.eq.b.or.(epsabs.le.0.0d+00.and
     *  .epsrel.lt.dmax1(0.5d+02*epmach,0.5d-28))) go to 999}
     
ier:= 6;
neval:= 0;
last:= 0;
alist[1]:= a;
blist[1]:= b;
rlist[1]:= 0.0;
elist[1]:= 0.0;
iord[1]:= 0;
result:= 0.0;
abserr:= 0.0;
if ((c=a)or(c=b)or((epsabs<=0.0)and(epsrel<dmax1(50*epmach,0.5E-28)))) then
   begin
   {writeln('a=',a,'  b=',b,'  c=',c, '  epsabs=',epsabs,'  epmach=',epmach);}
   exit;
   end;

{c
c           first approximation to the integral
c           -----------------------------------
c
      aa=a
      bb=b
      if (a.le.b) go to 10
      aa=b
      bb=a
10    ier=0
      krule = 1
      call dqc25c(f,aa,bb,c,result,abserr,krule,neval)
      last = 1
      rlist(1) = result
      elist(1) = abserr
      iord(1) = 1
      alist(1) = a
      blist(1) = b}
      
aa:= a;
bb:= b;
if a>b then
   begin
   aa:=b;
   bb:=a;
   end;
ier:= 0;
krule:= 1;
dqc25c(f, p, aa, bb, c, result, abserr, krule, neval);
last:= 1;
rlist[1]:= result;
elist[1]:= abserr;
iord[1]:= 1;
alist[1]:= a;
blist[1]:= b;

{c
c           test on accuracy
c
      errbnd = dmax1(epsabs,epsrel*dabs(result))
      if(limit.eq.1) ier = 1
      if(abserr.lt.dmin1(0.1d-01*dabs(result),errbnd)
     *  .or.ier.eq.1) go to 70}
errbnd:= dmax1(epsabs, epsrel*abs(result));
if limit=1 then ier:= 1;
if (abserr<dmin1(0.1E-1*abs(result),errbnd)) or (ier=1) then
   begin
   if aa=b then result:= - result;
   exit;
   end;
   
{c
c           initialization
c           --------------
c
      alist(1) = aa
      blist(1) = bb
      rlist(1) = result
      errmax = abserr
      maxerr = 1
      area = result
      errsum = abserr
      nrmax = 1
      iroff1 = 0
      iroff2 = 0}
alist[1]:= aa;
blist[1]:= bb;
rlist[1]:= result;
errmax:= abserr;
maxerr:= 1;
area:= result;
errsum:= abserr;
nrmax:= 1;
iroff1:= 0;
iroff2:= 0;

{c
c           main do-loop
c           ------------
c
      do 40 last = 2,limit}
last:= 2;
while last<=limit do
   begin
      
{c
c           bisect the subinterval with nrmax-th largest
c           error estimate.
c
        a1 = alist(maxerr)
        b1 = 0.5d+00*(alist(maxerr)+blist(maxerr))
        b2 = blist(maxerr)
        if(c.le.b1.and.c.gt.a1) b1 = 0.5d+00*(c+b2)
        if(c.gt.b1.and.c.lt.b2) b1 = 0.5d+00*(a1+c)
        a2 = b1
        krule = 2
        call dqc25c(f,a1,b1,c,area1,error1,krule,nev)
        neval = neval+nev
        call dqc25c(f,a2,b2,c,area2,error2,krule,nev)
        neval = neval+nev}
   a1:= alist[maxerr];
   b1:= 0.5*(alist[maxerr]+blist[maxerr]);
   b2:= blist[maxerr];
   if (c<=b1) and (c>a1) then b1:= 0.5*(c+b2);
   if (c>b1) and (c<b2) then b1:= 0.5*(a1+c);
   a2:= b1;
   krule:= 2;
   dqc25c(f, p, a1, b1, c, area1, error1, krule, nev);
   neval:= neval+nev;
   dqc25c(f, p, a2, b2, c, area2, error2, krule, nev);
   neval:= neval+nev;

{c
c           improve previous approximations to integral
c           and error and test for accuracy.
c
        area12 = area1+area2
        erro12 = error1+error2
        errsum = errsum+erro12-errmax
        area = area+area12-rlist(maxerr)
        if(dabs(rlist(maxerr)-area12).lt.0.1d-04*dabs(area12)
     *    .and.erro12.ge.0.99d+00*errmax.and.krule.eq.0)
     *    iroff1 = iroff1+1
        if(last.gt.10.and.erro12.gt.errmax.and.krule.eq.0)
     *    iroff2 = iroff2+1
        rlist(maxerr) = area1
        rlist(last) = area2
        errbnd = dmax1(epsabs,epsrel*dabs(area))
        if(errsum.le.errbnd) go to 15}
   area12:= area1+area2;
   erro12:= error1+error2;
   errsum:= errsum+erro12-errmax;
   area:= area+area12-rlist[maxerr];
   if (abs(rlist[maxerr]-area12)<0.1E-4*abs(area12)) and (erro12>=0.99*errmax)
      and (krule=0) then iroff1:= iroff1+1;
   if (last>10) and (erro12>errmax) and (krule=0) then iroff2:= iroff2+1;
   rlist[maxerr]:= area1;
   rlist[last]:= area2;
   errbnd:= dmax1(epsabs, epsrel*abs(area));
   if errsum>errbnd then
      begin
{c
c           test for roundoff error and eventually set error flag.
c
        if(iroff1.ge.6.and.iroff2.gt.20) ier = 2
c
c           set error flag in the case that number of interval
c           bisections exceeds limit.
c
        if(last.eq.limit) ier = 1
c
c           set error flag in the case of bad integrand behaviour
c           at a point of the integration range.
c
        if(dmax1(dabs(a1),dabs(b2)).le.(0.1d+01+0.1d+03*epmach)
     *    *(dabs(a2)+0.1d+04*uflow)) ier = 3}
      if (iroff1>=6) and (iroff2>20) then ier:=2;
      if last=limit then ier:=1;
      if dmax1(abs(a1),abs(b2))<=(1.0+100.0*epmach)*(abs(a2)+1000.0*uflow) then
         ier:= 3;
      end;

{c
c           append the newly-created intervals to the list.
c
   15   if(error2.gt.error1) go to 20
        alist(last) = a2
        blist(maxerr) = b1
        blist(last) = b2
        elist(maxerr) = error1
        elist(last) = error2
        go to 30
   20   alist(maxerr) = a2
        alist(last) = a1
        blist(last) = b1
        rlist(maxerr) = area2
        rlist(last) = area1
        elist(maxerr) = error2
        elist(last) = error1}
   if error2<=error1 then
      begin
      alist[last]:= a2;
      blist[maxerr]:= b1;
      blist[last]:= b2;
      elist[maxerr]:= error1;
      elist[last]:= error2;
      end
   else
      begin
      alist[maxerr]:= a2;
      alist[last]:= a1;
      blist[last]:= b1;
      rlist[maxerr]:= area2;
      rlist[last]:= area1;
      elist[maxerr]:= error2;
      elist[last]:= error1;
      end;
{c
c           call subroutine dqpsrt to maintain the descending ordering
c           in the list of error estimates and select the subinterval
c           with nrmax-th largest error estimate (to be bisected next).
c
   30    call dqpsrt(limit,last,maxerr,errmax,elist,iord,nrmax)
c ***jump out of do-loop
        if(ier.ne.0.or.errsum.le.errbnd) go to 50
   40 continue}
   dqpsrt(limit,last,maxerr,errmax,elist,iord,nrmax);
   if (ier<>0) or (errsum<=errbnd) then break;
   last:= last+1;
   end;
   
{c
c           compute final result.
c           ---------------------
c
   50 result = 0.0d+00
      do 60 k=1,last
        result = result+rlist(k)
   60 continue
      abserr = errsum
   70 if (aa.eq.b) result=-result
  999 return
      end}
result:= 0.0;
for k:=1 to last do
   result:= result+ rlist[k];
abserr:= errsum;
if aa=b then result:= -result;
end;


procedure dqawc(f: integfunc; p:pointer; a,b,c,epsabs,epsrel: double;
                var result,abserr: double; var neval,ier: integer;
                limit,lenw: integer; var last: integer;
		var iwork: nintvec; var work: nvec);
{c***begin prologue  dqawc
c***date written   800101   (yymmdd)
c***revision date  830518   (yymmdd)
c***category no.  h2a2a1,j4
c***keywords  automatic integrator, special-purpose,
c             cauchy principal value,
c             clenshaw-curtis, globally adaptive
c***author  piessens,robert ,appl. math. & progr. div. - k.u.leuven
c           de doncker,elise,appl. math. & progr. div. - k.u.leuven
c***purpose  the routine calculates an approximation result to a
c            cauchy principal value i = integral of f*w over (a,b)
c            (w(x) = 1/((x-c), c.ne.a, c.ne.b), hopefully satisfying
c            following claim for accuracy
c            abs(i-result).le.max(epsabe,epsrel*abs(i)).
c***description
c
c        computation of a cauchy principal value
c        standard fortran subroutine
c        double precision version
c
c
c        parameters
c         on entry
c            f      - double precision
c                     function subprogram defining the integrand
c                     function f(x). the actual name for f needs to be
c                     declared e x t e r n a l in the driver program.
c
c            a      - double precision
c                     under limit of integration
c
c            b      - double precision
c                     upper limit of integration
c
c            c      - parameter in the weight function, c.ne.a, c.ne.b.
c                     if c = a or c = b, the routine will end with
c                     ier = 6 .
c
c            epsabs - double precision
c                     absolute accuracy requested
c            epsrel - double precision
c                     relative accuracy requested
c                     if  epsabs.le.0
c                     and epsrel.lt.max(50*rel.mach.acc.,0.5d-28),
c                     the routine will end with ier = 6.
c
c         on return
c            result - double precision
c                     approximation to the integral
c
c            abserr - double precision
c                     estimate or the modulus of the absolute error,
c                     which should equal or exceed abs(i-result)
c
c            neval  - integer
c                     number of integrand evaluations
c
c            ier    - integer
c                     ier = 0 normal and reliable termination of the
c                             routine. it is assumed that the requested
c                             accuracy has been achieved.
c                     ier.gt.0 abnormal termination of the routine
c                             the estimates for integral and error are
c                             less reliable. it is assumed that the
c                             requested accuracy has not been achieved.
c            error messages
c                     ier = 1 maximum number of subdivisions allowed
c                             has been achieved. one can allow more sub-
c                             divisions by increasing the value of limit
c                             (and taking the according dimension
c                             adjustments into account). however, if
c                             this yields no improvement it is advised
c                             to analyze the integrand in order to
c                             determine the integration difficulties.
c                             if the position of a local difficulty
c                             can be determined (e.g. singularity,
c                             discontinuity within the interval) one
c                             will probably gain from splitting up the
c                             interval at this point and calling
c                             appropriate integrators on the subranges.
c                         = 2 the occurrence of roundoff error is detec-
c                             ted, which prevents the requested
c                             tolerance from being achieved.
c                         = 3 extremely bad integrand behaviour occurs
c                             at some points of the integration
c                             interval.
c                         = 6 the input is invalid, because
c                             c = a or c = b or
c                             (epsabs.le.0 and
c                              epsrel.lt.max(50*rel.mach.acc.,0.5d-28))
c                             or limit.lt.1 or lenw.lt.limit*4.
c                             result, abserr, neval, last are set to
c                             zero. exept when lenw or limit is invalid,
c                             iwork(1), work(limit*2+1) and
c                             work(limit*3+1) are set to zero, work(1)
c                             is set to a and work(limit+1) to b.
c
c         dimensioning parameters
c            limit - integer
c                    dimensioning parameter for iwork
c                    limit determines the maximum number of subintervals
c                    in the partition of the given integration interval
c                    (a,b), limit.ge.1.
c                    if limit.lt.1, the routine will end with ier = 6.
c
c           lenw   - integer
c                    dimensioning parameter for work
c                    lenw must be at least limit*4.
c                    if lenw.lt.limit*4, the routine will end with
c                    ier = 6.
c
c            last  - integer
c                    on return, last equals the number of subintervals
c                    produced in the subdivision process, which
c                    determines the number of significant elements
c                    actually in the work arrays.
c
c         work arrays
c            iwork - integer
c                    vector of dimension at least limit, the first k
c                    elements of which contain pointers
c                    to the error estimates over the subintervals,
c                    such that work(limit*3+iwork(1)), ... ,
c                    work(limit*3+iwork(k)) form a decreasing
c                    sequence, with k = last if last.le.(limit/2+2),
c                    and k = limit+1-last otherwise
c
c            work  - double precision
c                    vector of dimension at least lenw
c                    on return
c                    work(1), ..., work(last) contain the left
c                     end points of the subintervals in the
c                     partition of (a,b),
c                    work(limit+1), ..., work(limit+last) contain
c                     the right end points,
c                    work(limit*2+1), ..., work(limit*2+last) contain
c                     the integral approximations over the subintervals,
c                    work(limit*3+1), ..., work(limit*3+last)
c                     contain the error estimates.
c
c***references  (none)
c***routines called  dqawce,xerror
c***end prologue  dqawc}
var
{c
      double precision a,abserr,b,c,epsabs,epsrel,f,result,work
      integer ier,iwork,last,lenw,limit,lvl,l1,l2,l3,neval
c
      dimension iwork(limit),work(lenw)
c
      external f}
lvl, l1, l2, l3: integer;
work1, workl1, workl2, workl3: ^nvec;
{c
c         check validity of limit and lenw.
c
c***first executable statement  dqawc}
begin
{      ier = 6
      neval = 0
      last = 0
      result = 0.0d+00
      abserr = 0.0d+00
      if(limit.lt.1.or.lenw.lt.limit*4) go to 10}
ier:= 6;
neval:=0;
last:= 0;
result:=0.0;
abserr:= 0.0;
if (limit>=1) and (lenw>=limit*4) then
   begin
{c
c         prepare call for dqawce.
c
      l1 = limit+1
      l2 = limit+l1
      l3 = limit+l2
      call dqawce(f,a,b,c,epsabs,epsrel,limit,result,abserr,neval,ier,
     *  work(1),work(l1),work(l2),work(l3),iwork,last)
c
c         call error handler if necessary.
c
      lvl = 0}
   l1:= limit+1;
   l2:= limit+l1;
   l3:= limit+l2;
   work1:= @work[1];
   workl1:= @work[l1];
   workl2:= @work[l2];
   workl3:= @work[l3];
   dqawce(f, p, a, b, c, epsabs, epsrel, limit, result, abserr, neval, ier,
          work1^, workl1^, workl2^, workl3^, iwork, last);
   {writeln('Result from dqawce: ', result, '  IER= ',ier);}
   end;
   
{10    if(ier.eq.6) lvl = 1
      if(ier.ne.0) call xerror(26habnormal return from dqawc,26,ier,lvl)
      return
      end}
if ier=6 then lvl:=1;
{if ier<>0 then
   writeln('Abnormal return form dqawc. ERROR= ',ier,'  LEVEL= ',lvl);}
end;

end.
