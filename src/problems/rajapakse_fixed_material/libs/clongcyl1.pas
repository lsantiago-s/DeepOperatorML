unit clongcyl1;

interface
uses complex, besselj, besselz, axsgreen1;

{type
mat_const= record
   c11, c12, c33, c13, c44, rho, damp: double;
   end;}


const
longur=1;
longut=2;
longuz=3;
longsrr=4;
longstt=5;
longszz=6;
longsrt=7;
longstz=8;
longsrz=9;

function pzresponse0(mat: mat_const; omega, a, r: double; 
    component: integer):tcomplex;
    
function pzresponse1(mat: mat_const; omega, a, r: double; 
    component: integer):tcomplex;

function prresponse0(mat: mat_const; omega, a, r: double; 
    component: integer):tcomplex;
    
function prresponse1(mat: mat_const; omega, a, r: double; 
    component: integer):tcomplex;

function ptresponse0(mat: mat_const; omega, a, r: double; 
    component: integer):tcomplex;
    
function ptresponse1(mat: mat_const; omega, a, r: double; 
    component: integer):tcomplex;
    
implementation
type
matrix4= array[1..4,1..4] of tcomplex;
vector4= array[1..4] of tcomplex;

function besselj2(x: double): double;
begin
if x=0.0 then
   besselj2:=0.0
else
   besselj2:= 2.0/x * besselj1(x) - besselj0(x);
end;

function bessely2(x: double): double;
begin
if x=0.0 then
   bessely2:=0.0
else
   bessely2:= 2.0/x * bessely1(x) - bessely0(x);
end;

function besselj3(x: double): double;
begin
if x=0.0 then
   besselj3:=0.0
else
   besselj3:= 4.0/x * besselj2(x) - besselj1(x);
end;

function bessely3(x: double): double;
begin
if x=0.0 then
   bessely3:=0.0
else
   bessely3:= 4.0/x * bessely2(x) - bessely1(x);
end;

function pzresponse0(mat: mat_const; omega, a, r: double; 
    component: integer):tcomplex;
var
delta: double;
resp: tcomplex;

begin
delta:= omega*sqrt(mat.rho/mat.c44);

case component of
longuz:
   if r<a then
      resp:= -c_i*pi*a/2.0/mat.c44*besselj0(delta*r)*
          (besselj0(delta*a)-c_i*bessely0(delta*a))
   else
      resp:= -c_i*pi*a/2.0/mat.c44*besselj0(delta*a)*
          (besselj0(delta*r)-c_i*bessely0(delta*r));
	  
longsrz:
   if r<a then
      resp:= c_i*pi*delta*a/2.0*besselj1(delta*r)*
          (besselj0(delta*a)-c_i*bessely0(delta*a))
   else
      resp:= pi*delta*a/2.0*besselj0(delta*a)*
          (c_i*besselj1(delta*r)+bessely1(delta*r))
else
   resp:= czero;
end;{case}

pzresponse0:= resp;
end;

function pzresponse1(mat: mat_const; omega, a, r: double; 
    component: integer):tcomplex;
var
delta: double;
resp: tcomplex;

begin
delta:= omega*sqrt(mat.rho/mat.c44);

case component of
longuz:
   if r<a then
      resp:= -c_i*pi*a/2.0/mat.c44*besselj1(delta*r)*
          (besselj1(delta*a)-c_i*bessely1(delta*a))
   else
      resp:= -c_i*pi*a/2.0/mat.c44*besselj1(delta*a)*
          (besselj1(delta*r)-c_i*bessely1(delta*r));
	  
longsrz:
   if r<a then
      resp:= -c_i*pi*delta*a/4.0*(besselj0(delta*r)-besselj2(delta*r))*
          (besselj1(delta*a)-c_i*bessely1(delta*a))
   else
      resp:= pi*delta*a/4.0*besselj1(delta*a)*
          (-c_i*besselj0(delta*r)+c_i*besselj2(delta*r)
	   -bessely0(delta*r)+bessely2(delta*r));
longstz:
   if r<a then
      resp:= c_i*pi*a/2.0/r*besselj1(delta*r)*
          (besselj1(delta*a)-c_i*bessely1(delta*a))
   else
      resp:= c_i*pi*a/2.0/r*besselj1(delta*a)*
          (besselj1(delta*r)-c_i*bessely1(delta*r))
else
   resp:= czero;
end;{case}

pzresponse1:= resp;
end;

function urij(delta,r: double; beta, tau: tcomplex; m,i,j: integer): tcomplex;
var
b, t, fa, fb, bj, by, CJ0,CJ1,CY0,CY1,CH0,CH1: tcomplex;
d: double;

begin
d:= delta*r;
b:= d/csqrt(beta);
t:= d/csqrt(tau);

if m=0 then
   begin
   CJYHBS(t, 1, CJ0,CJ1,CY0,CY1,CH0,CH1);
   bj:= cj0;
   by:= cy0;
   CJYHBS(b, 1, CJ0,CJ1,CY0,CY1,CH0,CH1);
   fa:= -2.0*cj1;
   fb:= -2.0*cy1;
   end
else
   begin
   CJYHBS(t, 1, CJ0,CJ1,CY0,CY1,CH0,CH1);
   bj:= cj1;
   by:= cy1;
   CJYHBS(b, 1, CJ0,CJ1,CY0,CY1,CH0,CH1);
   fa:= cj0 - 2.0/b*cj1 + cj0;
   fb:= cy0 - 2.0/b*cy1 + cy0;
   end;

if i=1 then
   if j=1 then
      urij:= b*fa
   else
      urij:= 2.0*m*bj
else
   if j=1 then
      urij:= b*(fa-c_i*fb)
   else
      urij:= 2.0*m*(bj-c_i*by);
end;

function utij(delta,r: double; beta, tau: tcomplex; m,i,j: integer): tcomplex;
var
b, t, fa, fb, bj, by, CJ0,CJ1,CY0,CY1,CH0,CH1: tcomplex;
d: double;

begin
d:= delta*r;
b:= d/csqrt(beta);
t:= d/csqrt(tau);

if m=0 then
   begin
   CJYHBS(b, 1, CJ0,CJ1,CY0,CY1,CH0,CH1);
   bj:= cj0;
   by:= cy0;
   CJYHBS(t, 1, CJ0,CJ1,CY0,CY1,CH0,CH1);
   fa:= -2.0*cj1;
   fb:= -2.0*cy1;
   end
else
   begin
   CJYHBS(b, 1, CJ0,CJ1,CY0,CY1,CH0,CH1);
   bj:= cj1;
   by:= cy1;
   CJYHBS(t, 1, CJ0,CJ1,CY0,CY1,CH0,CH1);
   fa:= cj0 - 2.0/t*cj1 + cj0;
   fb:= cy0 - 2.0/t*cy1 + cy0;
   end;

if i=1 then
   if j=1 then
      utij:= -2.0*m*bj
   else
      utij:= -t*fa
else
   if j=1 then
      utij:= -2.0*m*(bj-c_i*by)
   else
      utij:= -t*(fa-c_i*fb);
end;

function srrij(delta,r: double; beta, tau: tcomplex; m,i,j: integer): tcomplex;
var
b, t, fa, fb, bj, by, CJ0,CJ1,CY0,CY1,CH0,CH1, ga, gb, p: tcomplex;
d: double;

begin
d:= delta*r;
b:= d/csqrt(beta);
t:= d/csqrt(tau);
if j=1 then p:=b else p:=t;

if m=0 then
   begin
   CJYHBS(p, 1, CJ0,CJ1,CY0,CY1,CH0,CH1);
   bj:= cj0;
   by:= cy0;
   fa:= -2.0*cj1;
   fb:= -2.0*cy1;
   ga:= 2.0*(2.0/p*cj1-cj0);
   gb:= 2.0*(2.0/p*cy1-cy0);
   end
else
   begin
   CJYHBS(p, 1, CJ0,CJ1,CY0,CY1,CH0,CH1);
   bj:= cj1;
   by:= cy1;
   fa:= cj0 - 2.0/p*cj1 + cj0;
   fb:= cy0 - 2.0/p*cy1 + cy0;
   ga:= -cj1 + (8.0/p/p - 1.0)*cj1 -4.0/p*cj0;
   gb:= -cy1 + (8.0/p/p - 1.0)*cy1 -4.0/p*cy0;
   end;

if i=1 then
   if j=1 then
      srrij:= sqr(d)*ga+2.0*(beta-2.0*tau)/csqrt(beta)*d*fa
              - (4.0*m*m*beta+2.0*d*d-8.0*m*m*tau)*bj
   else
      srrij:= 4.0*m*d*csqrt(tau)*fa - 8.0*m*tau*bj
else
   if j=1 then
      srrij:= sqr(d)*ga+2.0*d*csqrt(beta)*fa-(4.0*m*m*beta+2.0*d*d)*bj
              -c_i*d*d*gb-2.0*c_i*d*csqrt(beta)*fb+8.0*m*m*tau*(bj-c_i*by)
	      +2.0*c_i*(2.0*m*m*beta+d*d)*by-4.0*b*tau*(fa-c_i*fb)
   else
      srrij:= 4.0*m*d*csqrt(tau)*(fa-c_i*fb)-8.0*m*tau*(bj-c_i*by);
end;

function srtij(delta,r: double; beta, tau: tcomplex; m,i,j: integer): tcomplex;
var
b, t, fa, fb, bj, by, CJ0,CJ1,CY0,CY1,CH0,CH1, ga, gb, p: tcomplex;
d: double;

begin
d:= delta*r;
b:= d/csqrt(beta);
t:= d/csqrt(tau);
if j=1 then p:=b else p:=t;

if m=0 then
   begin
   CJYHBS(p, 1, CJ0,CJ1,CY0,CY1,CH0,CH1);
   bj:= cj0;
   by:= cy0;
   fa:= -2.0*cj1;
   fb:= -2.0*cy1;
   ga:= 2.0*(2.0/p*cj1-cj0);
   gb:= 2.0*(2.0/p*cy1-cy0);
   end
else
   begin
   CJYHBS(p, 1, CJ0,CJ1,CY0,CY1,CH0,CH1);
   bj:= cj1;
   by:= cy1;
   fa:= cj0 - 2.0/p*cj1 + cj0;
   fb:= cy0 - 2.0/p*cy1 + cy0;
   ga:= -cj1 + (8.0/p/p - 1.0)*cj1 -4.0/p*cj0;
   gb:= -cy1 + (8.0/p/p - 1.0)*cy1 -4.0/p*cy0;
   end;



{if i=1 then
   if j=1 then
      srtij:= 8.0*tau*m*bj-4.0*m*b*fa
   else
      srtij:= -csqr(t)*ga+2.0*t*fa+2.0*(t*t-2.0*m*m*tau)*bj
else
   if j=1 then
      srtij:= 8.0*tau*m*(bj-c_i*by)-4.0*m*b*(fa-c_i*fb)
   else
      srtij:= -csqr(t)*(ga-c_i*gb)+2.0*t*(fa-c_i*fb)
              +2.0*(t*t-2.0*m*m*tau)*(bj-c_i*by);}
	      
if i=1 then
   if j=1 then
      srtij:= 8.0*tau*m*bj-4.0*tau*m*b*fa
   else
      srtij:= -sqr(d)*ga+
         2.0*d*csqrt(tau)*fa+
	 2.0*(sqr(d)-2.0*m*m*tau)*bj
else
   if j=1 then
      srtij:= 8.0*tau*m*(bj-c_i*by)-4.0*tau*m*b*(fa-c_i*fb)
   else
      srtij:= -sqr(d)*(ga-c_i*gb)+
               2.0*csqrt(tau)*d*(fa-c_i*fb)
              +2.0*(sqr(d)-2.0*m*m*tau)*(bj-c_i*by);

end;

PROCEDURE linsolve(var a: matrix4; var b: vector4);
{ subrotina para resolver o sistema de equacoes lineares
  por eliminacao de GAUSS com recurso de troca da posicao
  das linhas quando encontra coeficiente zero na diagonal

  A: matriz do sistema;
  B: originalmente contem os coeficientes independentes
     posteriormente contem os valores das incognitas
  N: numero real de equacoes }

CONST
  small= 1.0E-12;


VAR
  n,n1,k,k1,j,l,i: longint;
  c, c1: tcomplex;
  ok: boolean;

BEGIN
  n:=4;
  ok:=true;
  n1:=n-1;
  for k:=1 to n1 do
   if ok then
    begin
    k1:=k+1;
    c:=a[k,k];
    if (abs(c.x)<= small) and (abs(c.y)<=small) then
    {elemento da diagonal muito pequeno?}
      begin
      j:=k1;
      ok:=false;
      repeat
        if (abs(a[j,k].x)> small) or
                (abs(a[j,k].y)> small) then
        {tenta a troca de linhas}
          begin
          for l:=k to n do
            begin
            c:=a[k,l];
            a[k,l]:=a[j,l];
            a[j,l]:=c
            end;
          c:=b[k];
          b[k]:=b[j];
          b[j]:=c;
          OK:=true
          end;
          j:=j+1;
      until (OK) or (j>n);
      end;
      if  ok then  {caso consiga}
         begin
         c:=a[k,k];             {processo de eliminacao}
         for j:=k1 to n do
           a[k,j]:= a[k,j]/c;
         b[k]:= b[k]/c;
         for i:=k1 to n do
           begin
           c:=a[i,k];
           for j:=k1 to n do
             begin
             c1:= c*a[k,j];
             a[i,j]:= a[i,j] - c1;
             end;
           c1:= c*b[k];
           b[i]:= b[i] - c1;
           end
         end
    end;
    if ((abs(a[n,n].x)<=small) and (abs(a[n,n].y)<=small)) or
       (not ok) then
            writeln(' **** SINGULARITY IN ROW ',K:5)
    else
        begin
        b[n]:= b[n]/a[n,n];  {calculo da solucao do sistema}
        for l:=1 to n1 do
          begin
          k:=n-l;
          k1:=k+1;
          for j:=k1 to n do
            begin
            c:= a[k,j]*b[j];
            b[k]:= b[k] - c;
            end;
          end;
        end
END;

function prtresponse(mat: mat_const; omega, a, r: double; 
    component, dir, m: integer):tcomplex;
var
 delta, s: double;
am: matrix4;
bm: vector4;
i, j: integer;
beta, tau, resp: tcomplex;

begin
with mat do
   begin
   delta:= omega*sqrt(rho/c44);
   beta:= c11*(1.0+c_i*damp)/c44;
   tau:= (c11-c12)*(1.0+c_i*damp)/2.0/c44;
   end;
   
for i:=1 to 2 do for j:=1 to 2 do
  begin
  s:=1.0;
  if i=2 then s:= -1.0;
  am[1,2*(i-1)+j]:= s*urij(delta, a, beta, tau, m, i, j);
  am[2,2*(i-1)+j]:= s*utij(delta, a, beta, tau, m, i, j);
  am[3,2*(i-1)+j]:= s*srrij(delta, a, beta, tau, m, i, j);
  am[4,2*(i-1)+j]:= s*srtij(delta, a, beta, tau, m, i, j);
  end;

{for i:=1 to 4 do for j:=1 to 4 do
   writeln(am[i,j].x, '    ', am[i,j].y);
writeln;}
  
for i:=1 to 4 do bm[i]:= czero;
if dir=1 then bm[3]:= 4.0*sqr(a)/mat.c44
else bm[4]:= 4.0*sqr(a)/mat.c44;

{for i:=1 to 4 do writeln(bm[i].x, '   ',bm[i].y);
writeln;}

linsolve(am,bm);

{for i:=1 to 4 do writeln(bm[i].x, '   ',bm[i].y);
writeln;}

case component of
   longur:
      if r<a then
         resp:= (bm[1]*urij(delta, r, beta, tau, m, 1, 1)
	        +bm[2]*urij(delta, r, beta, tau, m, 1, 2))/2.0/r
      else
         resp:= (bm[3]*urij(delta, r, beta, tau, m, 2, 1)
	        +bm[4]*urij(delta, r, beta, tau, m, 2, 2))/2.0/r;
   longut:
      if r<a then
         resp:= (bm[1]*utij(delta, r, beta, tau, m, 1, 1)
	        +bm[2]*utij(delta, r, beta, tau, m, 1, 2))/2.0/r
      else
         resp:= (bm[3]*utij(delta, r, beta, tau, m, 2, 1)
	        +bm[4]*utij(delta, r, beta, tau, m, 2, 2))/2.0/r;
   longsrr:
      if r<a then
         resp:= mat.c44*(bm[1]*srrij(delta, r, beta, tau, m, 1, 1)
	        +bm[2]*srrij(delta, r, beta, tau, m, 1, 2))/4.0/r/r
      else
         resp:= mat.c44*(bm[3]*srrij(delta, r, beta, tau, m, 2, 1)
	        +bm[4]*srrij(delta, r, beta, tau, m, 2, 2))/4.0/r/r;
   longsrt:
      if r<a then
         resp:= mat.c44*(bm[1]*srtij(delta, r, beta, tau, m, 1, 1)
	        +bm[2]*srtij(delta, r, beta, tau, m, 1, 2))/4.0/r/r
      else
         resp:= mat.c44*(bm[3]*srtij(delta, r, beta, tau, m, 2, 1)
	        +bm[4]*srtij(delta, r, beta, tau, m, 2, 2))/4.0/r/r
   else resp:= czero;
   end;{case}
prtresponse:= resp;
end;


function prresponse0(mat: mat_const; omega, a, r: double; 
    component: integer):tcomplex;
begin
prresponse0:= prtresponse(mat, omega, a, r, component, 1, 0);
end;
    
function prresponse1(mat: mat_const; omega, a, r: double; 
    component: integer):tcomplex;
begin
prresponse1:= prtresponse(mat, omega, a, r, component, 1, 1);
end;

function ptresponse0(mat: mat_const; omega, a, r: double; 
    component: integer):tcomplex;
begin
ptresponse0:= prtresponse(mat, omega, a, r, component, 2, 0);
end;
    
function ptresponse1(mat: mat_const; omega, a, r: double; 
    component: integer):tcomplex;
begin
ptresponse1:= prtresponse(mat, omega, a, r, component, 2, 1);
end;

    
end.
