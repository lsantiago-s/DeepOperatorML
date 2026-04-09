unit dmach;
interface
function d1mach(i:integer):double;
function dmax1(c1, c2: double): double;
function dmin1(c1, c2: double): double;

implementation
const
{small: array[1..2] of longint = ( 0,    1048576);
large: array[1..2] of longint = (-1, 2146435071);
right: array[1..2] of longint = ( 0, 1017118720);
diver: array[1..2] of longint = ( 0, 1018167296);
log10: array[1..2] of longint = ( 1352628735, 1070810131);
small: array[1..10] of longint= ( 0,    1048576, -1, 2146435071,
                                  0, 1017118720,  0, 1018167296,
                                  1352628735, 1070810131);}

small: array[1..5] of double= (2.225074e-308, 1.797693e+308, 1.110223e-16, 2.220446e-16, 3.010300e-01);




function d1mach(i:integer):double;
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
end;

end.
