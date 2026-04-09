unit axskern1;
interface
uses complex, besselj, besselz;

const
uzz=1;
urz=2;
szzz=3;
srzz=4;
srrz=5;
uzr=6;
urr=7;
szzr=8;
srzr=9;
srrr=10;
specialsrzz=100;
realpart=1;
imagpart=2;
pointload=1;
ringload=2;
diskload=3;
cylinderload=4;
anularload=5;
longcylinderload=6;
loadinfullspace=1;
loadoverhalfspace=2;
loadinhalfspace=3;
freesurfinfluence=10;
normalmode=0;
wobessel=1;
wobessel_b=2;
woload=3;
woload_b=4;
varyimag=5;

type

taxskerndata= record
alpha, beta, gamma, tau, kappa: tcomplex;
h,z,r,delta,loadradius,loadheight, aux1, aux2: double;
bvptype, loadtype, component, part, mode: integer;
end;

function axskernfunc(p: pointer; zeta: double): double;

function axskernsrzz0(p: pointer; zeta: double): double;

implementation
type
tcomplex2= array[1..2] of tcomplex;
tcomplex4= array[1..4] of tcomplex;
tcomplex6= array[1..6] of tcomplex;

function loadinfullspacesolution(z, l, delta: double;
         xi, omega: tcomplex2; eta: tcomplex6;
         component: integer): tcomplex;
var
zexp, a1, a2: tcomplex2;
result: tcomplex;

begin

if l=-1.0 then {point, ring or disk load}
   begin
   zexp[1]:= cexp(-delta*abs(z)*xi[1]);
   zexp[2]:= cexp(-delta*abs(z)*xi[2]);
   end
else {cylinder load}
   if abs(z)>=l then
      begin
      a1[1]:= -delta*(abs(z)+l)*xi[1];
      a2[1]:= -delta*(abs(z)-l)*xi[1];
      a1[2]:= -delta*(abs(z)+l)*xi[2];
      a2[2]:= -delta*(abs(z)-l)*xi[2];
      zexp[1]:= (cexp(a2[1])-cexp(a1[1]))/(delta*xi[1]);
      zexp[2]:= (cexp(a2[2])-cexp(a1[2]))/(delta*xi[2]);
      end
   else
      begin
      a1[1]:= -delta*(z+l)*xi[1];
      a2[1]:= -delta*(-z+l)*xi[1];
      a1[2]:= -delta*(z+l)*xi[2];
      a2[2]:= -delta*(-z+l)*xi[2];
      case component of
      urz,szzz,srrz,uzr,srzr:
         begin
         zexp[1]:= (-cexp(a1[1])+cexp(a2[1]))/(delta*xi[1]);
         zexp[2]:= (-cexp(a1[2])+cexp(a2[2]))/(delta*xi[2]);
	 if z<0.0 then
	    begin
	    zexp[1]:= -zexp[1];
	    zexp[2]:= -zexp[2];
	    end;
	 end;
      else if (component=specialsrzz) then
         begin
         zexp[1]:= (-cexp(a1[1])-cexp(a2[1]))/(delta*xi[1]);
         zexp[2]:= (-cexp(a1[2])-cexp(a2[2]))/(delta*xi[2]);
	 end
      else
         begin
         zexp[1]:= (2.0-cexp(a1[1])-cexp(a2[1]))/(delta*xi[1]);
         zexp[2]:= (2.0-cexp(a1[2])-cexp(a2[2]))/(delta*xi[2]);
	 end;
      end;{case}
      end;

case component of
uzz,urr:  result:= omega[1]*zexp[1]-omega[2]*zexp[2];
urz,uzr:  result:= zexp[1]-zexp[2];
szzz,szzr: result:= eta[5]*zexp[1]-eta[6]*zexp[2];
srzz,srzr, specialsrzz: result:= eta[3]*zexp[1]-eta[4]*zexp[2];
srrz,srrr: result:= eta[1]*zexp[1]-eta[2]*zexp[2];
end{case};

loadinfullspacesolution:= result;
end;

function freesurf_influence(z, l, delta: double;
         xi, omega, hexp: tcomplex2; eta: tcomplex6;
         component: integer): tcomplex;
var
k,zexp, lexp: tcomplex2;
result, a, b: tcomplex;

begin
zexp[1]:= cexp(-delta*z*xi[1]);
zexp[2]:= cexp(-delta*z*xi[2]);

if l=-1.0 then
   begin
   k[1]:= (eta[3]*eta[6]+eta[4]*eta[5])*hexp[1] -
          2.0*eta[4]*eta[6]*hexp[2];
   k[2]:= (eta[3]*eta[6]+eta[4]*eta[5])*hexp[2] -
          2.0*eta[3]*eta[5]*hexp[1];
   end
else
   begin
   lexp[1]:= cexp(-delta*l*xi[1]);
   lexp[2]:= cexp(-delta*l*xi[2]);
   {writeln(xi[1].x:10:5, xi[1].y:10:5,xi[2].x:10:5, xi[2].y:10:5);}
   a:= eta[3]*eta[6]+eta[4]*eta[5];
   b:= -2.0*eta[4]*eta[6];
   if abs(lexp[1].x)+abs(lexp[1].y)> 1.0E-10 then
      k[1]:= -a/(delta*xi[1])*hexp[1]*(lexp[1]-1.0/lexp[1])
   else
      k[1]:=czero;
   if abs(lexp[2].x)+abs(lexp[2].y)> 1.0E-10 then
      k[1]:= k[1] -b/(delta*xi[2])*hexp[2]*(lexp[2]-1.0/lexp[2]);

   b:=a;
   a:= -2.0*eta[3]*eta[5];
   if abs(lexp[1].x)+abs(lexp[1].y)> 1.0E-10 then
      k[2]:= -a/(delta*xi[1])*hexp[1]*(lexp[1]-1.0/lexp[1])
   else
      k[2]:=czero;
   if abs(lexp[2].x)+abs(lexp[2].y)> 1.0E-10 then
      k[2]:= k[2] -b/(delta*xi[2])*hexp[2]*(lexp[2]-1.0/lexp[2]);
   end;

case component of
uzz,urr:  result:= k[1]*omega[1]*zexp[1]+k[2]*omega[2]*zexp[2];
urz,uzr:  result:= k[1]*zexp[1]+k[2]*zexp[2];
szzz,szzr: result:= k[1]*eta[5]*zexp[1]+k[2]*eta[6]*zexp[2];
srzz,srzr, specialsrzz: result:= k[1]*eta[3]*zexp[1]+k[2]*eta[4]*zexp[2];
srrz,srrr: result:= k[1]*eta[1]*zexp[1]+k[2]*eta[2]*zexp[2];
end{case};

freesurf_influence:= result;
end;

procedure applyload(loadtype, component: integer;
          loadradius, loadh, delta, zeta: double; var result: tcomplex);
var
a,b, a1, a2, b1, b2: double;
begin
a:= delta*zeta*loadradius;
b:= loadradius/delta/zeta;
   case component of
   uzz,urz,szzz,srzz,srrz, specialsrzz:
      case loadtype of
      pointload: result:= result/2.0/pi;
      ringload, cylinderload:  result:= result*loadradius*besselj0(a);
      diskload:  result:= result*b*besselj1(a);
      anularload:
         begin
	 a1:= delta*zeta*(loadradius+loadh);
	 a2:= delta*zeta*(loadradius-loadh);
	 b1:= (loadradius+loadh)/delta/zeta;
	 b2:= (loadradius-loadh)/delta/zeta;
	 if abs(a2)<1.0E-8 then 
	    result:= result*b1*besselj1(a1)
	 else
	    result:= result*(b1*besselj1(a1)-b2*besselj1(a2));
	   
	 end;
      end;{case}
   else
      case loadtype of
      pointload: result:= czero;
      ringload, cylinderload:  result:= result*loadradius*besselj1(a);
      diskload:  result:= result*(b*pi/2.0)*
        (besselj1(a)*strvh0(a)-besselj0(a)*strvh1(a));
      anularload:
         begin
	 a1:= delta*zeta*(loadradius+loadh);
	 a2:= delta*zeta*(loadradius-loadh);
	 b1:= (loadradius+loadh)/delta/zeta;
	 b2:= (loadradius-loadh)/delta/zeta;
	 if abs(a2)<1.0E-8 then 
	    result:= result*(b1*pi/2.0)*
               (besselj1(a1)*strvh0(a1)-besselj0(a1)*strvh1(a1))
	 else
	    result:= result*((b1*pi/2.0)*
               (besselj1(a1)*strvh0(a1)-besselj0(a1)*strvh1(a1))-
	       (b2*pi/2.0)*
               (besselj1(a2)*strvh0(a2)-besselj0(a2)*strvh1(a2)));
	 end;
      end;{case}
   end{case};
end;

function axskernfunc(p: pointer; zeta: double): double;
const
verysmall= 1.0E-100;

var
s,lh: double;
data: ^taxskerndata;
phi, j0, j1, result, aux1, aux2, Q, ri: tcomplex;
xi, omega, zexp, zexph: tcomplex2;
ab: tcomplex4;
eta: tcomplex6;

begin

data:=p;
if zeta=1.0 then
   zeta:=1.000001;
with data^ do
   begin
   lh:= loadheight;
   if loadtype <> cylinderload then lh:=-1.0;
   j0:=1.0;
   j1:=1.0;
   if (mode<>wobessel) and (mode<>wobessel_b) and (mode<>woload_b) then
      begin
      j0:= besselj0(delta*zeta*r);
      j1:= besselj1(delta*zeta*r);
      end;
   phi:= csqr(gamma*sqr(zeta)-1.0-alpha) -
         4.0*alpha*(beta*sqr(sqr(zeta))-(beta+1.0)*sqr(zeta)+1.0);
   phi:= csqrt(phi);
   xi[1]:= csqrt((gamma*sqr(zeta)-1.0-alpha+phi)/(2.0*alpha));
   xi[2]:= csqrt((gamma*sqr(zeta)-1.0-alpha-phi)/(2.0*alpha));

   {if (xi[1].x<0.0) or (xi[2].x<0.0) then
      writeln('zeta= ', zeta:10:5);}

   case component of
   uzr,urr,szzr,srzr,srrr:
      begin
      omega[1]:= (alpha*csqr(xi[1])-sqr(zeta)+1.0)/(zeta*kappa*xi[1]);
      omega[2]:= (alpha*csqr(xi[2])-sqr(zeta)+1.0)/(zeta*kappa*xi[2]);
      eta[3]:= -xi[1]*omega[1]-zeta;
      eta[4]:= -xi[2]*omega[2]-zeta;
      eta[5]:= zeta*(kappa-1.0)*omega[1]-alpha*xi[1];
      eta[6]:= zeta*(kappa-1.0)*omega[2]-alpha*xi[2];
      if  (component=srrr) then if r<>0.0 then
         case mode of
         wobessel:
            begin
            {eta[1]:= omega[1]*(beta*zeta-(kappa-1.0));
            eta[2]:= omega[2]*(beta*zeta-(kappa-1.0));}
	       eta[1]:= beta*omega[1]*zeta-(kappa-1.0)*xi[1];
	       eta[2]:= beta*omega[2]*zeta-(kappa-1.0)*xi[2];
            end;
         wobessel_b:
            begin
            {eta[1]:= -2.0*tau*xi[1]*(1.0/delta/r);
            eta[2]:= -2.0*tau*xi[2]*(1.0/delta/r);}
	       eta[1]:= -2.0*tau*omega[1]/delta/r;
	       eta[2]:= -2.0*tau*omega[2]/delta/r;
            end;
         else
            begin
            {eta[1]:= omega[1]*(beta*zeta-(kappa-1.0))*j0 -
                2.0*tau*xi[1]*(j1/delta/r);
            eta[2]:= omega[2]*(beta*zeta-(kappa-1.0))*j0 -
                2.0*tau*xi[2]*(j1/delta/r);}
	       eta[1]:= (beta*omega[1]*zeta-(kappa-1.0)*xi[1])*j0+
	                (-2.0*tau*omega[1]/delta/r)*j1;
	       eta[2]:= (beta*omega[2]*zeta-(kappa-1.0)*xi[2])*j0+
	                (-2.0*tau*omega[2]/delta/r)*j1;
            end;
         end{case}
      else
         begin
         {eta[1]:= omega[1]*(beta*zeta-(kappa-1.0))*j0;
         eta[2]:= omega[2]*(beta*zeta-(kappa-1.0))*j0;}
	    eta[1]:= (omega[1]*beta*zeta - (kappa-1.0)*xi[1])*j0;
	    eta[2]:= (omega[2]*beta*zeta - (kappa-1.0)*xi[2])*j0;
         end;
      end;
   else
      begin
      omega[1]:= (beta*sqr(zeta)-csqr(xi[1])-1.0)/(zeta*kappa*xi[1]);
      omega[2]:= (beta*sqr(zeta)-csqr(xi[2])-1.0)/(zeta*kappa*xi[2]);
      eta[3]:= -xi[1] - zeta*omega[1];
      eta[4]:= -xi[2] - zeta*omega[2];
      eta[5]:= zeta*(kappa-1.0)-alpha*xi[1]*omega[1];
      eta[6]:= zeta*(kappa-1.0)-alpha*xi[2]*omega[2];
      if  (component=srrz) then if r<>0.0 then
         case mode of
         wobessel:
            begin
            eta[1]:= beta*zeta-(kappa-1.0)*xi[1]*omega[1];
            eta[2]:= beta*zeta-(kappa-1.0)*xi[2]*omega[2];
            end;
         wobessel_b:
            begin
            eta[1]:= -2.0*tau*(1.0/delta/r);
            eta[2]:= -2.0*tau*(1.0/delta/r);
            end;
         else
            begin
            eta[1]:= (beta*zeta-(kappa-1.0)*xi[1]*omega[1])*j0 -
                2.0*tau*(j1/delta/r);
            eta[2]:= (beta*zeta-(kappa-1.0)*xi[2]*omega[2])*j0 -
                2.0*tau*(j1/delta/r);
            end;
         end{case}
      else
         begin
         eta[1]:= (beta*zeta-omega[1]*xi[1]*(kappa-1.0))*j0;
         eta[2]:= (beta*zeta-omega[2]*xi[2]*(kappa-1.0))*j0;
         end;
      end;
   end{case};

   end;

with data^ do if bvptype=loadinfullspace then
   begin
   case component of
   uzz,urz,szzz,srzz,srrz, specialsrzz: q:= eta[6]-eta[5];
   else
      q:= eta[4]-eta[3]; {=-q}
   end;{case}
   if (abs(z)<0.0001) and ((component=szzz) or (component=srzr)) and
      (loadtype<>cylinderload) then
      result:= czero
   else
      result:= loadinfullspacesolution(z, lh, delta, xi, omega, eta,
               component);
   result:= result/q;

   case component of
   uzz,uzr,szzz,szzr: result:= result*j0*zeta;
   urz,urr,srzz,srzr,specialsrzz: result:= result*j1*zeta;
   else result:= result*zeta;
   end{case};

   if (mode<> woload) and (mode<>woload_b) then
      applyload(loadtype, component, loadradius, loadheight, delta, zeta, result);

   if z<0.0 then case component of
      urz,szzz,srrz,uzr,srzr: result:= -result;
      end;
  
   if abs(result.x)<verysmall then result.x:= 0.0;
   if abs(result.y)<verysmall then result.y:= 0.0;
   
   if part=realpart then
      axskernfunc:= result.x
   else
      axskernfunc:= result.y;
   end
else
   begin
   s:=1.0;
   case component of
   uzz,urz,szzz,srzz,srrz, specialsrzz: q:= eta[6]-eta[5];
   else
      begin
      s:=-1.0;
      q:= eta[4]-eta[3]; {=-q}
      end;
   end;{case}

   ri:= eta[3]*eta[6]-eta[4]*eta[5];

   zexp[1]:= cexp(-delta*xi[1]*h);
   zexp[2]:= cexp(-delta*xi[2]*h);

   if ((abs(z-h)<0.0001) and ((component=szzz) or (component=srzr))
      and (loadtype<>cylinderload)) or (bvptype=freesurfinfluence) then
      result:=0.0
   else
      result:= loadinfullspacesolution(z-h, lh, delta, xi, omega,
                eta, component);
   if z<h then case component of
      urz,szzz,srrz,uzr,srzr: result:= -result;
      end;
   result:= result-
        s*freesurf_influence(z,lh,delta,xi,omega,zexp,eta,component)/ri;
   result:= result/q;

   {result:= 2.0*(omega[1]*eta[4]-omega[2]*eta[3])/ri;}

   case component of
   uzz,uzr,szzz,szzr: result:= result*j0*zeta;
   urz,urr,srzz,srzr, specialsrzz: result:= result*j1*zeta;
   else result:= result*zeta;
   end{case};

   if (mode<> woload) and (mode<>woload_b) then
      applyload(loadtype, component, loadradius, loadheight, delta, zeta, result);

   if abs(result.x)<verysmall then result.x:= 0.0;
   if abs(result.y)<verysmall then result.y:= 0.0;

   if part=realpart then
      axskernfunc:= result.x
   else
      axskernfunc:= result.y;
   end;

end;

function axskernsrzz0(p: pointer; zeta: double): double;
{const
dv= 1.0E-5;}
var
lh: double;
data: ^taxskerndata;
phi, result, Q, ri, zt: tcomplex;
xi, omega, lexp, zexp: tcomplex2;
eta: tcomplex6;

bj, by, bh: array[0..1] of tcomplex;

begin

data:=p;
{writeln('zeta= ',zeta);}
{zt:= zeta+dv*c_i*zeta;
if zeta>1.0 then
   zt:= zeta + dv/sqr(zeta)*c_i;}
   
with data^ do
   begin
   if mode=varyimag then
      begin
      zt.x:= aux1;
      zt.y:= zeta;
      end
   else
      begin
      zt.x:= zeta;
      zt.y:= aux1;
      end;
   lh:=loadheight;
   ri:= delta*loadradius*zt;
   CJYHBS(ri, 1, bj[0], bj[1], by[0], by[1], bh[0], bh[1]);

   phi:= csqr(gamma*csqr(zt)-1.0-alpha) -
         4.0*alpha*(beta*csqr(csqr(zt))-(beta+1.0)*csqr(zt)+1.0);
   phi:= csqrt(phi);
   xi[1]:= csqrt((gamma*csqr(zt)-1.0-alpha+phi)/(2.0*alpha));
   xi[2]:= csqrt((gamma*csqr(zt)-1.0-alpha-phi)/(2.0*alpha));


      omega[1]:= (beta*csqr(zt)-csqr(xi[1])-1.0)/(zt*kappa*xi[1]);
      omega[2]:= (beta*csqr(zt)-csqr(xi[2])-1.0)/(zt*kappa*xi[2]);
      eta[3]:= -xi[1] - zt*omega[1];
      eta[4]:= -xi[2] - zt*omega[2];
      eta[5]:= zt*(kappa-1.0)-alpha*xi[1]*omega[1];
      eta[6]:= zt*(kappa-1.0)-alpha*xi[2]*omega[2];

   end;

q:= eta[6]-eta[5];
lexp[1]:= cexp(-lh*data^.delta*xi[1]);
lexp[2]:= cexp(-lh*data^.delta*xi[2]);
result:= (eta[3]/xi[1]*lexp[1]-eta[4]/xi[2]*lexp[2]);
{result:= (eta[3]/xi[1]-eta[4]/xi[2])/q;}

with data^ do if bvptype=loadinhalfspace then
   begin
     ri:= eta[3]*eta[6]-eta[4]*eta[5];

   zexp[1]:= cexp(-delta*xi[1]*h);
   zexp[2]:= cexp(-delta*xi[2]*h);

   result:= result-
        delta/2.0*freesurf_influence(z,lh,delta,xi,omega,zexp,eta,srzz)/ri;
   end;

result:= result/q;

if (data^.mode=normalmode) or (data^.mode=varyimag) then
   result:= -result*data^.loadradius*zt*
          (bj[0])*(bj[1])
else
   result:= result/pi/data^.delta;
   
   
if data^.part=realpart then
   axskernsrzz0:= result.x
else
   axskernsrzz0:= result.y;
end;

end.
