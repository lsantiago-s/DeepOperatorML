library axsgrsce;

uses
{$ifdef unix}
cthreads,
{$endif}
complex, axskern1, axsgreen1, clongcyl1;

type 
dataptr = ^datarec;
datarec = record
	matd: mat_const;
	rd,zd,hd,loadrd,loadhd,omegad: double;
	bvpd, ltd, cod, partd: integer;
	resd: double;
	end{record};
	
threadvar
finishr, finishi: integer;

function thrf(thrp : pointer) : longint;
var
p1: dataptr;

begin
p1:= dataptr(thrp);
with p1^ do
   resd:= axsgreenfunc(matd, rd, zd, hd, loadrd, loadhd, omegad, 
          bvpd, ltd, cod, partd);
if p1^.partd= realpart then finishr:= 1 else finishi:=1;
end;



procedure axsanisgreen(var c11, c12, c13, c33, c44, dens, damp,
                       r, z, h, loadr, loadh, omega: double;
		       var bvptype, loadtype, component: longint;
		       var resultr, resulti: double); cdecl;
var
mat: mat_const;
bvp, lt, co, co1, p: integer;
res: double;
res1: tcomplex;
p1, p2: datarec;
hd1, hd2: {thandle}qword;
begin

bvp:= bvptype;
lt:= loadtype;
co:= component;

mat.c11:= c11;
mat.c12:= c12;
mat.c13:= c13;
mat.c33:= c33;
mat.c44:= c44;
mat.rho:= dens;
mat.damp:= damp;

with p1 do
   begin
   matd:= mat;
   bvpd:= bvp;
   ltd:= lt;
   cod:= co;
   rd:= r;
   zd:= z;
   hd:= h;
   loadrd:= loadr;
   loadhd:= loadh;
   omegad:= omega;
   end;

finishr:= 0;
finishi:= 0;
if (lt=cylinderload) and ((co=srzz)or(co=srrr))  and 
   (abs(r-loadr)<1.0E-10) and (abs(z-h)<1.0E-10) then
   begin
   if co=srzz then 
      res1:= pzresponse0(mat, omega, 0.999*r, r, longsrz)
   else
      res1:= prresponse0(mat, omega, 0.999*r, r, longsrr);
   res:= res1.x - 2.0*axsgreenfunc(mat, r, 101.0*loadh, 0.0, loadr, 100.0*loadh,
       omega, loadinfullspace, cylinderload, co, realpart);
   if bvp=loadinhalfspace then
      res:= res  +
         axsgreenfunc(mat, r, z, h, loadr, loadh, omega, freesurfinfluence,
	      cylinderload, co, realpart);
   {finishr:= 1;}
   end
else
   begin
   {p1.partd:= realpart;}
   res:= axsgreenfunc(mat, r, z, h, loadr, loadh, omega, bvp, lt, co,
   realpart);
   {BeginThread(@thrf,@p1, hd1);}
   end;

resultr:=res;
{p2:= p1;
p2.partd:= imagpart;}
res:= axsgreenfunc(mat, r, z, h, loadr, loadh, omega, bvp, lt, co, imagpart);
{BeginThread(@thrf,@p2, hd2);}

{if finishr=0 then waitforthreadterminate(hd1,10000);
if finishi=0 then waitforthreadterminate(hd2,10000);
resultr:= p1.resd;
resulti:= p2.resd;}
resulti:= res;   
end;

procedure axstest(var zeta, result: double); cdecl;
var
mat: mat_const;
data: taxskerndata;

begin

mat.c11:= 3.0;
mat.c12:= 1.0;
mat.c13:= 1.0;
mat.c33:= 3.0;
mat.c44:= 1.0;
mat.rho:= 1.0;
mat.damp:= 0.01;
with mat do  with data do
   begin
   alpha:= (c33+damp*c33*c_i)/c44;
   beta:=  (c11+damp*c11*c_i)/c44;
   kappa:= ((c13+damp*c13*c_i)+c44)/c44;
   tau:= (c11*(1.0+damp*c_i)-c12*(1.0+damp*c_i))/2.0/c44;
   gamma:= 1.0 + alpha*beta - kappa*kappa;
   delta:= sqrt(rho/c44)*2.0;
   end;

data.h:=0.0;
data.z:= 0.0;
data.r:= 0.5;
data.loadradius:= 0.5;
data.loadheight:= 0.25;
data.bvptype:= loadinfullspace;
data.loadtype:= cylinderload;
data.component:= srzz;
data.part:= realpart;
data.mode:= normalmode;
data.aux1:= 0.0;

   {result:= axsgreensrzz0(mat, 0.5, 0.0, 0.0, 0.5, 0.25, 2.0, loadinfullspace,
              cylinderload, srzz, realpart);}
result:= axskernsrzz0(@data,zeta);
end;

exports axsanisgreen, axstest;

end.
