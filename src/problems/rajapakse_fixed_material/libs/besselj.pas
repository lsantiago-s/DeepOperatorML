unit besselj;

interface
function besselj0(x:double):double;
function besselj1(x:double):double;
function bessely0(x:double):double;
function bessely1(x:double):double;
FUNCTION STRVH0(XVALUE: double): double;
FUNCTION STRVH1(XVALUE: double): double;

implementation
uses dmach;
const
firstinitds: boolean = true;
firstdcsevl: boolean = true;
firstd9b0mp: boolean = true;
firstdbesj0: boolean = true;

firstd9b1mp: boolean = true;
firstdbesj1: boolean = true;

firstdbesy0: boolean = true;
firstdbesy1: boolean = true;

type
longvec= array[1..100] of double;
longvecptr= ^longvec;
dvec= array[0..100] of double;
dvecptr= ^dvec;

var
onepl, xmax0, xmax1, xsml0, xsml1, xmin1, xsmly0, xsmly1, xminy1:double;
nbm0, nbt02, nbm02, nbth0, ntj0, nty0: integer;
nbm1, nbt12, nbm12, nbth1, ntj1, nty1: integer;


function max(x1, x2: double): double;
begin
if x2>x1 then max:= x2 else max:= x1;
end;

function initds(var os:array of double; nos: integer;
         eta: double):integer;
{C
C  Initialize the orthogonal series, represented by the array OS, so
C  that INITDS is the number of terms needed to insure the error is no
C  larger than ETA.  Ordinarily, ETA will be chosen to be one-tenth
C  machine precision.
C
C             Input Arguments --
C   OS     double precision array of NOS coefficients in an orthogonal
C          series.
C   NOS    number of coefficients in OS.
C   ETA    single precision scalar containing requested accuracy of
C          series.
C
C***REFERENCES  (NONE)
C***ROUTINES CALLED  XERMSG
C***REVISION HISTORY  (YYMMDD)
C   770601  DATE WRITTEN
C   890531  Changed all specific intrinsics to generic.  (WRB)
C   890831  Modified array declarations.  (WRB)
C   891115  Modified error message.  (WRB)
C   891115  REVISION DATE from Version 3.2
C   891214  Prologue converted to Version 4.0 format.  (BAB)
C   900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
C***END PROLOGUE  INITDS
      DOUBLE PRECISION OS(*)
C***FIRST EXECUTABLE STATEMENT  INITDS}
var
i, ii: integer;
err: double;

begin
initds:= 9999;

{      IF (NOS .LT. 1) CALL XERMSG ('SLATEC', 'INITDS',
     +   'Number of coefficients is less than 1', 2, 1)}
if nos<1 then
   begin
   writeln('INITDS: Number of coefficients is less than 1');
   exit;
   end;
{C
      ERR = 0.
      DO 10 II = 1,NOS
        I = NOS + 1 - II
        ERR = ERR + ABS(REAL(OS(I)))
        IF (ERR.GT.ETA) GO TO 20
   10 CONTINUE}
err:=0.0;
for ii:= 1 to nos do
   begin
   i:= nos+1-ii;
   err:= err + abs(os[i]);
   if err>eta then break;
   end;

{C
   20 IF (I .EQ. NOS) CALL XERMSG ('SLATEC', 'INITDS',
     +   'Chebyshev series too short for specified accuracy', 1, 1)
      INITDS = I
C
      RETURN
      END}
if i=nos then
   writeln('INITDS: Chebyshev series too short for specified accuracy');
initds:=i;
end;

function dcsevl(x:double; var cs: array of double;
         n:integer): double;
{C
C  Evaluate the N-term Chebyshev series CS at X.  Adapted from
C  a method presented in the paper by Broucke referenced below.
C
C       Input Arguments --
C  X    value at which the series is to be evaluated.
C  CS   array of N terms of a Chebyshev series.  In evaluating
C       CS, only half the first coefficient is summed.
C  N    number of terms in array CS.
C
C***REFERENCES  R. Broucke, Ten subroutines for the manipulation of
C                 Chebyshev series, Algorithm 446, Communications of
C                 the A.C.M. 16, (1973) pp. 254-256.
C               L. Fox and I. B. Parker, Chebyshev Polynomials in
C                 Numerical Analysis, Oxford University Press, 1968,
C                 page 56.
C***ROUTINES CALLED  D1MACH, XERMSG
C***REVISION HISTORY  (YYMMDD)
C   770401  DATE WRITTEN
C   890831  Modified array declarations.  (WRB)
C   890831  REVISION DATE from Version 3.2
C   891214  Prologue converted to Version 4.0 format.  (BAB)
C   900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
C   900329  Prologued revised extensively and code rewritten to allow
C           X to be slightly outside interval (-1,+1).  (WRB)
C   920501  Reformatted the REFERENCES section.  (WRB)
C***END PROLOGUE  DCSEVL}

{      DOUBLE PRECISION B0, B1, B2, CS(*), ONEPL, TWOX, X, D1MACH
      LOGICAL FIRST
      SAVE FIRST, ONEPL
      DATA FIRST /.TRUE./}
var
b0, b1, b2, twox: double;
i, ni: integer;

begin
{C***FIRST EXECUTABLE STATEMENT  DCSEVL
      IF (FIRST) ONEPL = 1.0D0 + D1MACH(4)
      FIRST = .FALSE.
      IF (N .LT. 1) CALL XERMSG ('SLATEC', 'DCSEVL',
     +   'NUMBER OF TERMS .LE. 0', 2, 2)
      IF (N .GT. 1000) CALL XERMSG ('SLATEC', 'DCSEVL',
     +   'NUMBER OF TERMS .GT. 1000', 3, 2)
      IF (ABS(X) .GT. ONEPL) CALL XERMSG ('SLATEC', 'DCSEVL',
     +   'X OUTSIDE THE INTERVAL (-1,+1)', 1, 1)}
if firstdcsevl then
   onepl:= 1.0 + d1mach(4);
firstdcsevl:= false;
{writeln('d1mach(4)= ', d1mach(4));}
if (n<1) or (n>1000) or (abs(x)>onepl) then
   begin
   writeln('DCSEVL: X outside the interval (-1,+1)');
   dcsevl:=0.0;
   exit;
   end;

{C
      B1 = 0.0D0
      B0 = 0.0D0
      TWOX = 2.0D0*X
      DO 10 I = 1,N
         B2 = B1
         B1 = B0
         NI = N + 1 - I
         B0 = TWOX*B1 - B2 + CS(NI)
   10 CONTINUE}
b1:=0.0;
b0:=0.0;
twox:= 2.0*x;
for i:= 1 to n do
   begin
   b2:= b1;
   b1:= b0;
   ni:= n+1-i;
   b0:= twox*b1 - b2 + cs[ni];
   end;

{C
      DCSEVL = 0.5D0*(B0-B2)
C
      RETURN
      END}
dcsevl:= 0.5*(b0-b2);
end;

procedure d9b0mp( x: double; var ampl, theta: double);
{C
C Evaluate the modulus and phase for the Bessel J0 and Y0 functions.
C
C Series for BM0        on the interval  1.56250E-02 to  6.25000E-02
C                                        with weighted error   4.40E-32
C                                         log weighted error  31.36
C                               significant figures required  30.02
C                                    decimal places required  32.14
C
C Series for BTH0       on the interval  0.          to  1.56250E-02
C                                        with weighted error   2.66E-32
C                                         log weighted error  31.57
C                               significant figures required  30.67
C                                    decimal places required  32.40
C
C Series for BM02       on the interval  0.          to  1.56250E-02
C                                        with weighted error   4.72E-32
C                                         log weighted error  31.33
C                               significant figures required  30.00
C                                    decimal places required  32.13
C
C Series for BT02       on the interval  1.56250E-02 to  6.25000E-02
C                                        with weighted error   2.99E-32
C                                         log weighted error  31.52
C                               significant figures required  30.61
C                                    decimal places required  32.32
C
C***REFERENCES  (NONE)
C***ROUTINES CALLED  D1MACH, DCSEVL, INITDS, XERMSG
C***REVISION HISTORY  (YYMMDD)
C   770701  DATE WRITTEN
C   890531  Changed all specific intrinsics to generic.  (WRB)
C   890531  REVISION DATE from Version 3.2
C   891214  Prologue converted to Version 4.0 format.  (BAB)
C   900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
C   900720  Routine changed from user-callable to subsidiary.  (WRB)
C   920618  Removed space from variable names.  (RWC, WRB)
C***END PROLOGUE  D9B0MP}

{      DOUBLE PRECISION X, AMPL, THETA, BM0CS(37), BT02CS(39),
     1  BM02CS(40), BTH0CS(44), XMAX, PI4, Z, D1MACH, DCSEVL
      LOGICAL FIRST
      SAVE BM0CS, BTH0CS, BM02CS, BT02CS, PI4, NBM0, NBT02,
     1 NBM02, NBTH0, XMAX, FIRST}

const
bm0cs: array [0..37] of double = (0.0,
+0.9211656246827742712573767730182E-1,
-0.1050590997271905102480716371755E-2,
+0.1470159840768759754056392850952E-4,
-0.5058557606038554223347929327702E-6,
+0.2787254538632444176630356137881E-7,
-0.2062363611780914802618841018973E-8,
+0.1870214313138879675138172596261E-9,
-0.1969330971135636200241730777825E-10,
+0.2325973793999275444012508818052E-11,
-0.3009520344938250272851224734482E-12,
+0.4194521333850669181471206768646E-13,
-0.6219449312188445825973267429564E-14,
+0.9718260411336068469601765885269E-15,
-0.1588478585701075207366635966937E-15,
+0.2700072193671308890086217324458E-16,
-0.4750092365234008992477504786773E-17,
+0.8615128162604370873191703746560E-18,
-0.1605608686956144815745602703359E-18,
+0.3066513987314482975188539801599E-19,
-0.5987764223193956430696505617066E-20,
+0.1192971253748248306489069841066E-20,
-0.2420969142044805489484682581333E-21,
+0.4996751760510616453371002879999E-22,
-0.1047493639351158510095040511999E-22,
+0.2227786843797468101048183466666E-23,
-0.4801813239398162862370542933333E-24,
+0.1047962723470959956476996266666E-24,
-0.2313858165678615325101260800000E-25,
+0.5164823088462674211635199999999E-26,
-0.1164691191850065389525401599999E-26,
+0.2651788486043319282958336000000E-27,
-0.6092559503825728497691306666666E-28,
+0.1411804686144259308038826666666E-28,
-0.3298094961231737245750613333333E-29,
+0.7763931143074065031714133333333E-30,
-0.1841031343661458478421333333333E-30,
+0.4395880138594310737100799999999E-31);

BTH0CS: array[0..44] of double = ( 0.0,
-0.24901780862128936717709793789967E+0,
+0.48550299609623749241048615535485E-3,
-0.54511837345017204950656273563505E-5,
+0.13558673059405964054377445929903E-6,
-0.55691398902227626227583218414920E-8,
+0.32609031824994335304004205719468E-9,
-0.24918807862461341125237903877993E-10,
+0.23449377420882520554352413564891E-11,
-0.26096534444310387762177574766136E-12,
+0.33353140420097395105869955014923E-13,
-0.47890000440572684646750770557409E-14,
+0.75956178436192215972642568545248E-15,
-0.13131556016891440382773397487633E-15,
+0.24483618345240857495426820738355E-16,
-0.48805729810618777683256761918331E-17,
+0.10327285029786316149223756361204E-17,
-0.23057633815057217157004744527025E-18,
+0.54044443001892693993017108483765E-19,
-0.13240695194366572724155032882385E-19,
+0.33780795621371970203424792124722E-20,
-0.89457629157111779003026926292299E-21,
+0.24519906889219317090899908651405E-21,
-0.69388422876866318680139933157657E-22,
+0.20228278714890138392946303337791E-22,
-0.60628500002335483105794195371764E-23,
+0.18649748964037635381823788396270E-23,
-0.58783732384849894560245036530867E-24,
+0.18958591447999563485531179503513E-24,
-0.62481979372258858959291620728565E-25,
+0.21017901684551024686638633529074E-25,
-0.72084300935209253690813933992446E-26,
+0.25181363892474240867156405976746E-26,
-0.89518042258785778806143945953643E-27,
+0.32357237479762298533256235868587E-27,
-0.11883010519855353657047144113796E-27,
+0.44306286907358104820579231941731E-28,
-0.16761009648834829495792010135681E-28,
+0.64292946921207466972532393966088E-29,
-0.24992261166978652421207213682763E-29,
+0.98399794299521955672828260355318E-30,
-0.39220375242408016397989131626158E-30,
+0.15818107030056522138590618845692E-30,
-0.64525506144890715944344098365426E-31,
+0.26611111369199356137177018346367E-31);

BM02CS: array [0..40] of double= ( 0.0,
+0.9500415145228381369330861335560E-1,
-0.3801864682365670991748081566851E-3,
+0.2258339301031481192951829927224E-5,
-0.3895725802372228764730621412605E-7,
+0.1246886416512081697930990529725E-8,
-0.6065949022102503779803835058387E-10,
+0.4008461651421746991015275971045E-11,
-0.3350998183398094218467298794574E-12,
+0.3377119716517417367063264341996E-13,
-0.3964585901635012700569356295823E-14,
+0.5286111503883857217387939744735E-15,
-0.7852519083450852313654640243493E-16,
+0.1280300573386682201011634073449E-16,
-0.2263996296391429776287099244884E-17,
+0.4300496929656790388646410290477E-18,
-0.8705749805132587079747535451455E-19,
+0.1865862713962095141181442772050E-19,
-0.4210482486093065457345086972301E-20,
+0.9956676964228400991581627417842E-21,
-0.2457357442805313359605921478547E-21,
+0.6307692160762031568087353707059E-22,
-0.1678773691440740142693331172388E-22,
+0.4620259064673904433770878136087E-23,
-0.1311782266860308732237693402496E-23,
+0.3834087564116302827747922440276E-24,
-0.1151459324077741271072613293576E-24,
+0.3547210007523338523076971345213E-25,
-0.1119218385815004646264355942176E-25,
+0.3611879427629837831698404994257E-26,
-0.1190687765913333150092641762463E-26,
+0.4005094059403968131802476449536E-27,
-0.1373169422452212390595193916017E-27,
+0.4794199088742531585996491526437E-28,
-0.1702965627624109584006994476452E-28,
+0.6149512428936330071503575161324E-29,
-0.2255766896581828349944300237242E-29,
+0.8399707509294299486061658353200E-30,
-0.3172997595562602355567423936152E-30,
+0.1215205298881298554583333026514E-30,
-0.4715852749754438693013210568045E-31);

BT02CS: array[0..39] of double = ( 0.0,
-0.24548295213424597462050467249324E+0 ,
+0.12544121039084615780785331778299E-2 ,
-0.31253950414871522854973446709571E-4 ,
+0.14709778249940831164453426969314E-5 ,
-0.99543488937950033643468850351158E-7 ,
+0.85493166733203041247578711397751E-8 ,
-0.86989759526554334557985512179192E-9 ,
+0.10052099533559791084540101082153E-9 ,
-0.12828230601708892903483623685544E-10,
+0.17731700781805131705655750451023E-11,
-0.26174574569485577488636284180925E-12,
+0.40828351389972059621966481221103E-13,
-0.66751668239742720054606749554261E-14,
+0.11365761393071629448392469549951E-14,
-0.20051189620647160250559266412117E-15,
+0.36497978794766269635720591464106E-16,
-0.68309637564582303169355843788800E-17,
+0.13107583145670756620057104267946E-17,
-0.25723363101850607778757130649599E-18,
+0.51521657441863959925267780949333E-19,
-0.10513017563758802637940741461333E-19,
+0.21820381991194813847301084501333E-20,
-0.46004701210362160577225905493333E-21,
+0.98407006925466818520953651199999E-22,
-0.21334038035728375844735986346666E-22,
+0.46831036423973365296066286933333E-23,
-0.10400213691985747236513382399999E-23,
+0.23349105677301510051777740800000E-24,
-0.52956825323318615788049749333333E-25,
+0.12126341952959756829196287999999E-25,
-0.28018897082289428760275626666666E-26,
+0.65292678987012873342593706666666E-27,
-0.15337980061873346427835733333333E-27,
+0.36305884306364536682359466666666E-28,
-0.86560755713629122479172266666666E-29,
+0.20779909972536284571238399999999E-29,
-0.50211170221417221674325333333333E-30,
+0.12208360279441714184191999999999E-30,
-0.29860056267039913454250666666666E-31);

PI4 = 0.785398163397448309615660845819876E0;

var
eta, z: double;

begin
{      DATA FIRST /.TRUE./
C***FIRST EXECUTABLE STATEMENT  D9B0MP}

{      IF (FIRST) THEN
         ETA = 0.1*REAL(D1MACH(3))
         NBM0 = INITDS (BM0CS, 37, ETA)
         NBT02 = INITDS (BT02CS, 39, ETA)
         NBM02 = INITDS (BM02CS, 40, ETA)
         NBTH0 = INITDS (BTH0CS, 44, ETA)}
if firstd9b0mp then
   begin
   {writeln('firstD9B0MP=true');}
   eta:= 0.1*d1mach(3);
   nbm0:= initds(bm0cs, 37, eta);
   nbt02:= initds(bt02cs, 39, eta);
   nbm02:= initds(bm02cs, 40, eta);
   nbth0:= initds(bth0cs, 44, eta);

{C
         XMAX = 1.0D0/D1MACH(4)
      ENDIF}
   xmax0:= 1.0/d1mach(4);
   end
{else writeln('firstD9B0MP=false')};

{      FIRST = .FALSE.}
firstd9b0mp:= false;

{C
      IF (X .LT. 4.D0) CALL XERMSG ('SLATEC', 'D9B0MP',
     +   'X MUST BE GE 4', 1, 2)}
if x<4.0 then 
   begin
   writeln('D9B0MP: x must be >= 4');
   exit;
   end;

{C
      IF (X.GT.8.D0) GO TO 20
      Z = (128.D0/(X*X) - 5.D0)/3.D0
      AMPL = (.75D0 + DCSEVL (Z, BM0CS, NBM0))/SQRT(X)
      THETA = X - PI4 + DCSEVL (Z, BT02CS, NBT02)/X
      RETURN}
if x<=8.0 then
   begin
   z:= (128.0/(x*x) -5.0)/3.0;
   ampl:= (0.75 + dcsevl(z, bm0cs, nbm0))/sqrt(x);
   theta:= x - pi4 + dcsevl(z, bt02cs, nbt02)/x;
   end
else
{C
 20   IF (X .GT. XMAX) CALL XERMSG ('SLATEC', 'D9B0MP',
     +   'NO PRECISION BECAUSE X IS BIG', 2, 2)}
   begin
   if x>xmax0 then 
      begin
      writeln('D9B0MP: no precision because x is big');
      exit;
      end;
{C
      Z = 128.D0/(X*X) - 1.D0
      AMPL = (.75D0 + DCSEVL (Z, BM02CS, NBM02))/SQRT(X)
      THETA = X - PI4 + DCSEVL (Z, BTH0CS, NBTH0)/X
      RETURN}
   z:= 128.0/(x*x) - 1.0;
   ampl:= (0.75+dcsevl(z, bm02cs, nbm02))/sqrt(x);
   theta:= x - pi4 + dcsevl(z, bth0cs, nbth0)/x;
   end;
{C
      END}
end;

function besselj0(x:double):double;
{C
C DBESJ0(X) calculates the double precision Bessel function of
C the first kind of order zero for double precision argument X.
C
C Series for BJ0        on the interval  0.          to  1.60000E+01
C                                        with weighted error   4.39E-32
C                                         log weighted error  31.36
C                               significant figures required  31.21
C                                    decimal places required  32.00
C
C***REFERENCES  (NONE)
C***ROUTINES CALLED  D1MACH, D9B0MP, DCSEVL, INITDS
C***REVISION HISTORY  (YYMMDD)
C   770701  DATE WRITTEN
C   890531  Changed all specific intrinsics to generic.  (WRB)
C   890531  REVISION DATE from Version 3.2
C   891214  Prologue converted to Version 4.0 format.  (BAB)
C***END PROLOGUE  DBESJ0}

{      DOUBLE PRECISION X, BJ0CS(19), AMPL, THETA, XSML, Y, D1MACH,
     1  DCSEVL
      LOGICAL FIRST
      SAVE BJ0CS, NTJ0, XSML, FIRST}
const
BJ0CS: array[0..19] of double = (0.0,
+0.10025416196893913701073127264074E+0,
-0.66522300776440513177678757831124E+0,
+0.24898370349828131370460468726680E+0,
-0.33252723170035769653884341503854E-1,
+0.23114179304694015462904924117729E-2,
-0.99112774199508092339048519336549E-4,
+0.28916708643998808884733903747078E-5,
-0.61210858663032635057818407481516E-7,
+0.98386507938567841324768748636415E-9,
-0.12423551597301765145515897006836E-10,
+0.12654336302559045797915827210363E-12,
-0.10619456495287244546914817512959E-14,
+0.74706210758024567437098915584000E-17,
-0.44697032274412780547627007999999E-19,
+0.23024281584337436200523093333333E-21,
-0.10319144794166698148522666666666E-23,
+0.40608178274873322700800000000000E-26,
-0.14143836005240913919999999999999E-28,
+0.43910905496698880000000000000000E-31);

var
ampl, theta, y: double;

{DATA FIRST /.TRUE./
C***FIRST EXECUTABLE STATEMENT  DBESJ0}
begin
{      IF (FIRST) THEN
         NTJ0 = INITDS (BJ0CS, 19, 0.1*REAL(D1MACH(3)))
         XSML = SQRT(8.0D0*D1MACH(3))
      ENDIF
      FIRST = .FALSE.}
if firstdbesj0 then
   begin
   {writeln('firstdbesj0 = true');}
   ntj0:= initds(bj0cs, 19, 0.1*d1mach(3));
   xsml0:= sqrt(8.0*d1mach(3));
   end
{else writeln('firstdbesj0 = false')};
firstdbesj0:= false;

{C
      Y = ABS(X)
      IF (Y.GT.4.0D0) GO TO 20
C
      DBESJ0 = 1.0D0
      IF (Y.GT.XSML) DBESJ0 = DCSEVL (.125D0*Y*Y-1.D0, BJ0CS, NTJ0)
      RETURN
C
 20   CALL D9B0MP (Y, AMPL, THETA)
      DBESJ0 = AMPL * COS(THETA)
C
      RETURN
      END}
y:= abs(x);
if y<=4.0 then
   begin
   besselj0:= 1.0;
   if y>xsml0 then
      besselj0:= dcsevl(0.125*y*y-1.0, bj0cs, ntj0);
   end
else
  begin
  d9b0mp(y,ampl,theta);
  besselj0:= ampl * cos(theta);
  end;
end;

procedure d9b1mp( x: double; var ampl, theta: double);
{C
C Evaluate the modulus and phase for the Bessel J1 and Y1 functions.
C
C Series for BM1        on the interval  1.56250E-02 to  6.25000E-02
C                                        with weighted error   4.91E-32
C                                         log weighted error  31.31
C                               significant figures required  30.04
C                                    decimal places required  32.09
C
C Series for BT12       on the interval  1.56250E-02 to  6.25000E-02
C                                        with weighted error   3.33E-32
C                                         log weighted error  31.48
C                               significant figures required  31.05
C                                    decimal places required  32.27
C
C Series for BM12       on the interval  0.          to  1.56250E-02
C                                        with weighted error   5.01E-32
C                                         log weighted error  31.30
C                               significant figures required  29.99
C                                    decimal places required  32.10
C
C Series for BTH1       on the interval  0.          to  1.56250E-02
C                                        with weighted error   2.82E-32
C                                         log weighted error  31.55
C                               significant figures required  31.12
C                                    decimal places required  32.37
C
C***SEE ALSO  DBESJ1, DBESY1
C***REFERENCES  (NONE)
C***ROUTINES CALLED  D1MACH, DCSEVL, INITDS, XERMSG
C***REVISION HISTORY  (YYMMDD)
C   770701  DATE WRITTEN
C   890531  Changed all specific intrinsics to generic.  (WRB)
C   890531  REVISION DATE from Version 3.2
C   891214  Prologue converted to Version 4.0 format.  (BAB)
C   900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
C   900720  Routine changed from user-callable to subsidiary.  (WRB)
C   920618  Removed space from variable name and code restructured to
C           use IF-THEN-ELSE.  (RWC, WRB)
C***END PROLOGUE  D9B1MP
      DOUBLE PRECISION X, AMPL, THETA, BM1CS(37), BT12CS(39),
     1  BM12CS(40), BTH1CS(44), XMAX, PI4, Z, D1MACH, DCSEVL
      LOGICAL FIRST
      SAVE BM1CS, BT12CS, BTH1CS, BM12CS, PI4, NBM1, NBT12,
     1 NBM12, NBTH1, XMAX, FIRST}

const
BM1CS: array[0..37] of double = (0.0,
+0.1069845452618063014969985308538E+0,
+0.3274915039715964900729055143445E-2,
-0.2987783266831698592030445777938E-4,
+0.8331237177991974531393222669023E-6,
-0.4112665690302007304896381725498E-7,
+0.2855344228789215220719757663161E-8,
-0.2485408305415623878060026596055E-9,
+0.2543393338072582442742484397174E-10,
-0.2941045772822967523489750827909E-11,
+0.3743392025493903309265056153626E-12,
-0.5149118293821167218720548243527E-13,
+0.7552535949865143908034040764199E-14,
-0.1169409706828846444166290622464E-14,
+0.1896562449434791571721824605060E-15,
-0.3201955368693286420664775316394E-16,
+0.5599548399316204114484169905493E-17,
-0.1010215894730432443119390444544E-17,
+0.1873844985727562983302042719573E-18,
-0.3563537470328580219274301439999E-19,
+0.6931283819971238330422763519999E-20,
-0.1376059453406500152251408930133E-20,
+0.2783430784107080220599779327999E-21,
-0.5727595364320561689348669439999E-22,
+0.1197361445918892672535756799999E-22,
-0.2539928509891871976641440426666E-23,
+0.5461378289657295973069619199999E-24,
-0.1189211341773320288986289493333E-24,
+0.2620150977340081594957824000000E-25,
-0.5836810774255685901920938666666E-26,
+0.1313743500080595773423615999999E-26,
-0.2985814622510380355332778666666E-27,
+0.6848390471334604937625599999999E-28,
-0.1584401568222476721192960000000E-28,
+0.3695641006570938054301013333333E-29,
-0.8687115921144668243012266666666E-30,
+0.2057080846158763462929066666666E-30,
-0.4905225761116225518523733333333E-31);

BT12CS: array[0..39] of double = (0.0,
+0.73823860128742974662620839792764E+0 ,
-0.33361113174483906384470147681189E-2 ,
+0.61463454888046964698514899420186E-4 ,
-0.24024585161602374264977635469568E-5 ,
+0.14663555577509746153210591997204E-6 ,
-0.11841917305589180567005147504983E-7 ,
+0.11574198963919197052125466303055E-8 ,
-0.13001161129439187449366007794571E-9 ,
+0.16245391141361731937742166273667E-10,
-0.22089636821403188752155441770128E-11,
+0.32180304258553177090474358653778E-12,
-0.49653147932768480785552021135381E-13,
+0.80438900432847825985558882639317E-14,
-0.13589121310161291384694712682282E-14,
+0.23810504397147214869676529605973E-15,
-0.43081466363849106724471241420799E-16,
+0.80202544032771002434993512550400E-17,
-0.15316310642462311864230027468799E-17,
+0.29928606352715568924073040554666E-18,
-0.59709964658085443393815636650666E-19,
+0.12140289669415185024160852650666E-19,
-0.25115114696612948901006977706666E-20,
+0.52790567170328744850738380799999E-21,
-0.11260509227550498324361161386666E-21,
+0.24348277359576326659663462400000E-22,
-0.53317261236931800130038442666666E-23,
+0.11813615059707121039205990399999E-23,
-0.26465368283353523514856789333333E-24,
+0.59903394041361503945577813333333E-25,
-0.13690854630829503109136383999999E-25,
+0.31576790154380228326413653333333E-26,
-0.73457915082084356491400533333333E-27,
+0.17228081480722747930705920000000E-27,
-0.40716907961286507941068800000000E-28,
+0.96934745136779622700373333333333E-29,
-0.23237636337765716765354666666666E-29,
+0.56074510673522029406890666666666E-30,
-0.13616465391539005860522666666666E-30,
+0.33263109233894654388906666666666E-31);

BM12CS: array [0..40] of double = ( 0.0,
+0.9807979156233050027272093546937E-1,
+0.1150961189504685306175483484602E-2,
-0.4312482164338205409889358097732E-5,
+0.5951839610088816307813029801832E-7,
-0.1704844019826909857400701586478E-8,
+0.7798265413611109508658173827401E-10,
-0.4958986126766415809491754951865E-11,
+0.4038432416421141516838202265144E-12,
-0.3993046163725175445765483846645E-13,
+0.4619886183118966494313342432775E-14,
-0.6089208019095383301345472619333E-15,
+0.8960930916433876482157048041249E-16,
-0.1449629423942023122916518918925E-16,
+0.2546463158537776056165149648068E-17,
-0.4809472874647836444259263718620E-18,
+0.9687684668292599049087275839124E-19,
-0.2067213372277966023245038117551E-19,
+0.4646651559150384731802767809590E-20,
-0.1094966128848334138241351328339E-20,
+0.2693892797288682860905707612785E-21,
-0.6894992910930374477818970026857E-22,
+0.1830268262752062909890668554740E-22,
-0.5025064246351916428156113553224E-23,
+0.1423545194454806039631693634194E-23,
-0.4152191203616450388068886769801E-24,
+0.1244609201503979325882330076547E-24,
-0.3827336370569304299431918661286E-25,
+0.1205591357815617535374723981835E-25,
-0.3884536246376488076431859361124E-26,
+0.1278689528720409721904895283461E-26,
-0.4295146689447946272061936915912E-27,
+0.1470689117829070886456802707983E-27,
-0.5128315665106073128180374017796E-28,
+0.1819509585471169385481437373286E-28,
-0.6563031314841980867618635050373E-29,
+0.2404898976919960653198914875834E-29,
-0.8945966744690612473234958242979E-30,
+0.3376085160657231026637148978240E-30,
-0.1291791454620656360913099916966E-30,
+0.5008634462958810520684951501254E-31);

BTH1CS: array[0..44] of double = ( 0.0,
+0.74749957203587276055443483969695E+0,
-0.12400777144651711252545777541384E-2,
+0.99252442404424527376641497689592E-5,
-0.20303690737159711052419375375608E-6,
+0.75359617705690885712184017583629E-8,
-0.41661612715343550107630023856228E-9,
+0.30701618070834890481245102091216E-10,
-0.28178499637605213992324008883924E-11,
+0.30790696739040295476028146821647E-12,
-0.38803300262803434112787347554781E-13,
+0.55096039608630904934561726208562E-14,
-0.86590060768383779940103398953994E-15,
+0.14856049141536749003423689060683E-15,
-0.27519529815904085805371212125009E-16,
+0.54550796090481089625036223640923E-17,
-0.11486534501983642749543631027177E-17,
+0.25535213377973900223199052533522E-18,
-0.59621490197413450395768287907849E-19,
+0.14556622902372718620288302005833E-19,
-0.37022185422450538201579776019593E-20,
+0.97763074125345357664168434517924E-21,
-0.26726821639668488468723775393052E-21,
+0.75453300384983271794038190655764E-22,
-0.21947899919802744897892383371647E-22,
+0.65648394623955262178906999817493E-23,
-0.20155604298370207570784076869519E-23,
+0.63417768556776143492144667185670E-24,
-0.20419277885337895634813769955591E-24,
+0.67191464220720567486658980018551E-25,
-0.22569079110207573595709003687336E-25,
+0.77297719892989706370926959871929E-26,
-0.26967444512294640913211424080920E-26,
+0.95749344518502698072295521933627E-27,
-0.34569168448890113000175680827627E-27,
+0.12681234817398436504211986238374E-27,
-0.47232536630722639860464993713445E-28,
+0.17850008478186376177858619796417E-28,
-0.68404361004510395406215223566746E-29,
+0.26566028671720419358293422672212E-29,
-0.10450402527914452917714161484670E-29,
+0.41618290825377144306861917197064E-30,
-0.16771639203643714856501347882887E-30,
+0.68361997776664389173535928028528E-31,
-0.28172247861233641166739574622810E-31);

PI4 = 0.785398163397448309615660845819876E0;
var
eta, z: double;


{      DATA FIRST /.TRUE./
C***FIRST EXECUTABLE STATEMENT  D9B1MP}
begin
{      IF (FIRST) THEN
         ETA = 0.1*REAL(D1MACH(3))
         NBM1 = INITDS (BM1CS, 37, ETA)
         NBT12 = INITDS (BT12CS, 39, ETA)
         NBM12 = INITDS (BM12CS, 40, ETA)
         NBTH1 = INITDS (BTH1CS, 44, ETA)
C
         XMAX = 1.0D0/D1MACH(4)
      ENDIF
      FIRST = .FALSE.}
if firstd9b1mp then
   begin
   eta:= 0.1*d1mach(3);
   nbm1:= initds(bm1cs, 37, eta);
   nbt12:= initds(bt12cs, 39, eta);
   nbm12:= initds(bm12cs, 40, eta);
   nbth1:= initds(bth1cs, 44, eta);

   xmax1:= 1.0/d1mach(4);
   end;
firstd9b1mp:= false;
{writeln('nbth1= ',nbth1);}


{C
      IF (X .LT. 4.0D0) THEN
         CALL XERMSG ('SLATEC', 'D9B1MP', 'X must be .GE. 4', 1, 2)
         AMPL = 0.0D0
         THETA = 0.0D0}
if x<4.0 then
   begin
   ampl:=0.0;
   theta:=0.0;
   end
{      ELSE IF (X .LE. 8.0D0) THEN
         Z = (128.0D0/(X*X) - 5.0D0)/3.0D0
         AMPL = (0.75D0 + DCSEVL (Z, BM1CS, NBM1))/SQRT(X)
         THETA = X - 3.0D0*PI4 + DCSEVL (Z, BT12CS, NBT12)/X}
else if x<=8.0 then
   begin
   z:= (128.0/(x*x) -5.0)/3.0;
   ampl:= (0.75+ dcsevl(z,bm1cs,nbm1))/sqrt(x);
   theta:= x - 3.0*pi4 + dcsevl(z,bt12cs,nbt12)/x;
   end
{      ELSE
         IF (X .GT. XMAX) CALL XERMSG ('SLATEC', 'D9B1MP',
     +      'No precision because X is too big', 2, 2)
C
         Z = 128.0D0/(X*X) - 1.0D0
         AMPL = (0.75D0 + DCSEVL (Z, BM12CS, NBM12))/SQRT(X)
         THETA = X - 3.0D0*PI4 + DCSEVL (Z, BTH1CS, NBTH1)/X
      ENDIF
      RETURN
      END}
else
   begin
   if x>xmax1 then 
      begin
      writeln('D9B1MP: no precision because x is too big');
      exit;
      end;
   z:= 128.0/(x*x) - 1.0;
   ampl:= (0.75 + dcsevl(z, bm12cs, nbm12))/sqrt(x);
   theta:= x - 3.0*pi4 + dcsevl(z,bth1cs, nbth1)/x;
   end;
end;

function besselj1(x:double):double;
{C
C DBESJ1(X) calculates the double precision Bessel function of the
C first kind of order one for double precision argument X.
C
C Series for BJ1        on the interval  0.          to  1.60000E+01
C                                        with weighted error   1.16E-33
C                                         log weighted error  32.93
C                               significant figures required  32.36
C                                    decimal places required  33.57
C
C***REFERENCES  (NONE)
C***ROUTINES CALLED  D1MACH, D9B1MP, DCSEVL, INITDS, XERMSG
C***REVISION HISTORY  (YYMMDD)
C   780601  DATE WRITTEN
C   890531  Changed all specific intrinsics to generic.  (WRB)
C   890531  REVISION DATE from Version 3.2
C   891214  Prologue converted to Version 4.0 format.  (BAB)
C   900315  CALLs to XERROR changed to CALLs to XERMSG.  (THJ)
C   910401  Corrected error in code which caused values to have the
C           wrong sign for arguments less than 4.0.  (WRB)
C***END PROLOGUE  DBESJ1}

{      DOUBLE PRECISION X, BJ1CS(19), AMPL, THETA, XSML, XMIN, Y,
     1  D1MACH, DCSEVL
      LOGICAL FIRST
      SAVE BJ1CS, NTJ1, XSML, XMIN, FIRST}
const
BJ1CS: array [0..19] of double = ( 0.0,
-0.117261415133327865606240574524003E+0,
-0.253615218307906395623030884554698E+0,
+0.501270809844695685053656363203743E-1,
-0.463151480962508191842619728789772E-2,
+0.247996229415914024539124064592364E-3,
-0.867894868627882584521246435176416E-5,
+0.214293917143793691502766250991292E-6,
-0.393609307918317979229322764073061E-8,
+0.559118231794688004018248059864032E-10,
-0.632761640466139302477695274014880E-12,
+0.584099161085724700326945563268266E-14,
-0.448253381870125819039135059199999E-16,
+0.290538449262502466306018688000000E-18,
-0.161173219784144165412118186666666E-20,
+0.773947881939274637298346666666666E-23,
-0.324869378211199841143466666666666E-25,
+0.120223767722741022720000000000000E-27,
-0.395201221265134933333333333333333E-30,
+0.116167808226645333333333333333333E-32);

{DATA FIRST /.TRUE./}
var
y, ampl, theta: double;

begin
{C***FIRST EXECUTABLE STATEMENT  DBESJ1
      IF (FIRST) THEN
         NTJ1 = INITDS (BJ1CS, 19, 0.1*REAL(D1MACH(3)))
C
         XSML = SQRT(8.0D0*D1MACH(3))
         XMIN = 2.0D0*D1MACH(1)
      ENDIF
      FIRST = .FALSE.}
if firstdbesj1 then
   begin
   ntj1:= initds(bj1cs, 19, 0.1*d1mach(3));
   xsml1:= sqrt(8.0*d1mach(3));
   xmin1:= 2.0*d1mach(1);
   end;
firstdbesj1:= false;

{C
      Y = ABS(X)
      IF (Y.GT.4.0D0) GO TO 20}
y:= abs(x);
if y<= 4.0 then
   begin
{C
      DBESJ1 = 0.0D0
      IF (Y.EQ.0.0D0) RETURN
      IF (Y .LE. XMIN) CALL XERMSG ('SLATEC', 'DBESJ1',
     +   'ABS(X) SO SMALL J1 UNDERFLOWS', 1, 1)
      IF (Y.GT.XMIN) DBESJ1 = 0.5D0*X
      IF (Y.GT.XSML) DBESJ1 = X*(.25D0 + DCSEVL (.125D0*Y*Y-1.D0,
     1  BJ1CS, NTJ1) )
      RETURN}
   besselj1:= 0.0;
   if y=0.0 then 
      exit;
   if y<= xmin1 then 
      begin
      writeln('DBESJ1: abs(x) so small J1 underflows');
      exit;
      end
   else
      besselj1:= 0.5*x;
   if y>xsml1 then
      besselj1:= x*(0.25+dcsevl(0.125*y*y - 1.0, bj1cs, ntj1));
   end
else
   begin
{C
 20   CALL D9B1MP (Y, AMPL, THETA)
      DBESJ1 = SIGN (AMPL, X) * COS(THETA)
C
      RETURN
      END}
   d9b1mp(y, ampl, theta);
   if x>=0.0 then
      besselj1:= abs(ampl)*cos(theta)
   else
      besselj1:= -abs(ampl)*cos(theta);

   end;
end;

function bessely0(X:double): double;

{      double precision function dbesy0 (x)
c august 1980 edition.  w. fullerton, c3, los alamos scientific lab.
      double precision x, by0cs(19), ampl, theta, twodpi, xsml,
     1  y, alnhaf, d1mach, dcsevl, dbesj0, dlog, dsin, dsqrt
      external d1mach, dbesj0, dcsevl, dlog, dsin, dsqrt, initds
c
c series for by0        on the interval  0.          to  1.60000e+01
c                                        with weighted error   8.14e-32
c                                         log weighted error  31.09
c                               significant figures required  30.31
c                                    decimal places required  31.73
c}
const
by0cs: array[0..19] of double =( 0.0,
-0.1127783939286557321793980546028E-1, 
-0.1283452375604203460480884531838E+0, 
-0.1043788479979424936581762276618E+0, 
+0.2366274918396969540924159264613E-1, 
-0.2090391647700486239196223950342E-2, 
+0.1039754539390572520999246576381E-3, 
-0.3369747162423972096718775345037E-5, 
+0.7729384267670667158521367216371E-7, 
-0.1324976772664259591443476068964E-8, 
+0.1764823261540452792100389363158E-10,
-0.1881055071580196200602823012069E-12,
+0.1641865485366149502792237185749E-14,
-0.1195659438604606085745991006720E-16,
+0.7377296297440185842494112426666E-19,
-0.3906843476710437330740906666666E-21,
+0.1795503664436157949829120000000E-23,
-0.7229627125448010478933333333333E-26,
+0.2571727931635168597333333333333E-28,
-0.8141268814163694933333333333333E-31);

twodpi= 0.636619772367581343075535053490057E0;
alnhaf= -0.69314718055994530941723212145818E0;


{      data nty0, xsml / 0, 0.d0 /
c}


var
y, ampl, theta: double;

begin
{      if (nty0.ne.0) go to 10
      nty0 = initds (by0cs, 19, 0.1*sngl(d1mach(3)))
      xsml = dsqrt (4.0d0*d1mach(3))
c}
if firstdbesy0 then
   begin
   nty0:= initds(by0cs, 19, 0.1*d1mach(3));
   xsmly0:= sqrt(4.0*d1mach(3));
   end;
firstdbesy0:= false;

{ 10   if (x.le.0.d0) call seteru (29hdbesy0  x is zero or negative, 29,
     1  1, 2)}
     
     
{      if (x.gt.4.0d0) go to 20
c
      y = 0.d0
      if (x.gt.xsml) y = x*x
      dbesy0 = twodpi*(alnhaf+dlog(x))*dbesj0(x) + .375d0
     1  + dcsevl (.125d0*y-1.d0, by0cs, nty0)
      return
c}
if x<=4.0 then
   begin
   y:= 0.0;
   if x>xsmly0 then y:= sqr(x);
   bessely0:= twodpi*(alnhaf+ln(x))*besselj0(x)+0.375
              + dcsevl(0.125*y-1.0,by0cs,nty0);
   exit;
   end
else

{ 20   call d9b0mp (x, ampl, theta)
      dbesy0 = ampl * dsin(theta)
      return
c
      end}
   begin
   d9b0mp(x,ampl,theta);
   bessely0:= ampl* sin(theta);
   end;
end;


function bessely1(x: double): double;

{      double precision function dbesy1 (x)
c july 1977 edition.  w. fullerton, c3, los alamos scientific lab.
      double precision x, by1cs(20), ampl, theta, twodpi, xmin, xsml,
     1  y, d1mach, dcsevl, dbesj1, dexp, dlog, dsin, dsqrt
      external d1mach, dbesj1, dcsevl, dexp, dlog, dsin, dsqrt, initds
c
c series for by1        on the interval  0.          to  1.60000e+01
c                                        with weighted error   8.65e-33
c                                         log weighted error  32.06
c                               significant figures required  32.17
c                                    decimal places required  32.71
c}
const
by1cs: array[0..20] of double = (0.0,
+0.320804710061190862932352018628015E-1, 
+0.126270789743350044953431725999727E+1, 
+0.649996189992317500097490637314144E-2, 
-0.893616452886050411653144160009712E-1, 
+0.132508812217570954512375510370043E-1, 
-0.897905911964835237753039508298105E-3, 
+0.364736148795830678242287368165349E-4, 
-0.100137438166600055549075523845295E-5, 
+0.199453965739017397031159372421243E-7, 
-0.302306560180338167284799332520743E-9, 
+0.360987815694781196116252914242474E-11,
-0.348748829728758242414552947409066E-13,
+0.278387897155917665813507698517333E-15,
-0.186787096861948768766825352533333E-17,
+0.106853153391168259757070336000000E-19,
-0.527472195668448228943872000000000E-22,
+0.227019940315566414370133333333333E-24,
-0.859539035394523108693333333333333E-27,
+0.288540437983379456000000000000000E-29,
-0.864754113893717333333333333333333E-32);


Twodpi= 0.636619772367581343075535053490057;

{      data nty1, xmin, xsml / 0, 2*0.d0 /
c}
VAR
y, ampl, theta: double;

begin

{      if (nty1.ne.0) go to 10
      nty1 = initds (by1cs, 20, 0.1*sngl(d1mach(3)))
c
      xmin = 1.571d0 * dexp (dmax1(dlog(d1mach(1)), -dlog(d1mach(2))) +
     1  0.01d0)
      xsml = dsqrt (4.0d0*d1mach(3))
c}
if firstdbesy1 then
   begin
   nty1:= initds(by1cs, 20, 0.1*d1mach(3));
   xminy1:= 1.571*exp(max(d1mach(1),-ln(d1mach(2)))+0.01);
   xsmly1:= sqrt(4.0*d1mach(3));
   end;
firstdbesy1:= false;


{ 10   if (x.le.0.d0) call seteru (29hdbesy1  x is zero or negative, 29,
     1  1, 2)}
     
{      if (x.gt.4.0d0) go to 20
c
      if (x.lt.xmin) call seteru (31hdbesy1  x so small y1 overflows,
     1  31, 3, 2)
      y = 0.d0
      if (x.gt.xsml) y = x*x
      dbesy1 = twodpi * dlog(0.5d0*x)*dbesj1(x) + (0.5d0 +
     1  dcsevl (.125d0*y-1.d0, by1cs, nty1))/x
      return
c}
if x<= 4.0 then
   begin
   y:=0.0;
   if x>xsmly1 then y:= sqr(x);
   bessely1:= twodpi* ln(0.5*x)* besselj1(x) + (0.5+dcsevl(0.125*y-1.0,
               by1cs, nty1))/x;
   exit;
   end
else
   

 {20   call d9b1mp (x, ampl, theta)
      dbesy1 = ampl * dsin(theta)
      return
c
      end}
      
   begin
   d9b1mp(x,ampl,theta);
   bessely1:= ampl*sin(theta);
   end;
end;




FUNCTION CHEVAL(N:integer;A:dvecptr;T:double): double;
(*C
C   This function evaluates a Chebyshev series, using the
C   Clenshaw method with Reinsch modification, as analysed
C   in the paper by Oliver.
C
C   INPUT PARAMETERS
C
C       N - INTEGER - The no. of terms in the sequence
C
C       A - DOUBLE PRECISION ARRAY, dimension 0 to N - The coefficients of
C           the Chebyshev series
C
C       T - DOUBLE PRECISION - The value at which the series is to be
C           evaluated
C
C
C   REFERENCES
C
C        "An error analysis of the modified Clenshaw method for
C         evaluating Chebyshev and Fourier series" J. Oliver,
C         J.I.M.A., vol. 20, 1977, pp379-391
C
C
C MACHINE-DEPENDENT CONSTANTS: NONE
C
C
C INTRINSIC FUNCTIONS USED;
C
C    ABS
C
C
C AUTHOR:  Dr. Allan J. MacLeod,
C          Dept. of Mathematics and Statistics,
C          University of Paisley ,
C          High St.,
C          PAISLEY,
C          SCOTLAND
C
C
C LATEST MODIFICATION:   21 December , 1992
C
C *)
var
I: integer;
D1,D2,TT,U0,U1,U2: double;

const
ZERO: double=0.0;
HALF: double=0.5;
TEST: double=0.6;
TWO:  double=2.0;

begin
      U1 := ZERO;
{C
C   If ABS ( T )  < 0.6 use the standard Clenshaw method
C}
      IF ( ABS( T ) < TEST ) THEN
         begin
         U0 := ZERO;
         TT := T + T;
         for I := N downto 0 do
            begin
            U2 := U1;
            U1 := U0;
            U0 := TT * U1 + A^[ I ] - U2;
            end;
         CHEVAL :=  ( U0 - U2 ) / TWO;
         end
      ELSE
{C
C   If ABS ( T )  > =  0.6 use the Reinsch modification
C}
         begin
         D1 := ZERO;
{C
C   T > =  0.6 code
C}
         IF ( T > ZERO ) THEN
            begin
            TT :=  ( T - HALF ) - HALF;
            TT := TT + TT;
            for I := N downto 0 do
               begin
               D2 := D1;
               U2 := U1;
               D1 := TT * U2 + A^[ I ] + D2;
               U1 := D1 + U2;
               end;
            CHEVAL :=  ( D1 + D2 ) / TWO;
            end
         ELSE
            begin
{C
C   T < =  -0.6 code
C}
            TT :=  ( T + HALF ) + HALF;
            TT := TT + TT;
            for I := N downto 0 do
               begin
               D2 := D1;
               U2 := U1;
               D1 := TT * U2 + A^[ I ] - D2;
               U1 := D1 - U2;
               end;
            CHEVAL :=  ( D1 - D2 ) / TWO;
            END;
         END;
      END;


FUNCTION STRVH0(XVALUE: double): double;
(*C
C
C   DESCRIPTION:
C
C      This function calculates the value of the Struve function
C      of order 0, denoted H0(x), for the argument XVALUE, defined
C
C         STRVHO(x) = (2/pi) integral{0 to pi/2} sin(x cos(t)) dt
C
C      H0 also satisfies the second-order equation
C
C                 x*D(Df) + Df + x*f = 2x/pi
C
C      The code uses Chebyshev expansions whose coefficients are
C      given to 20D.
C
C
C   ERROR RETURNS:
C
C      As the asymptotic expansion of H0 involves the Bessel function
C      of the second kind Y0, there is a problem for large x, since
C      we cannot accurately calculate the value of Y0. An error message
C      is printed and STRVH0 returns the value 0.0.
C
C
C   MACHINE-DEPENDENT CONSTANTS:
C
C      NTERM1 - The no. of terms to be used in the array ARRH0. The
C               recommended value is such that
C                      ABS(ARRH0(NTERM1)) < EPS/100.
C
C      NTERM2 - The no. of terms to be used in the array ARRH0A. The
C               recommended value is such that
C                      ABS(ARRH0A(NTERM2)) < EPS/100.
C
C      NTERM3 - The no. of terms to be used in the array AY0ASP. The
C               recommended value is such that
C                      ABS(AY0ASP(NTERM3)) < EPS/100.
C
C      NTERM4 - The no. of terms to be used in the array AY0ASQ. The
C               recommended value is such that
C                      ABS(AY0ASQ(NTERM4)) < EPS/100.
C
C      XLOW - The value for which H0(x) = 2*x/pi to machine precision, if
C             abs(x) < XLOW. The recommended value is
C                      XLOW = 3 * SQRT(EPSNEG)
C
C      XHIGH - The value above which we are unable to calculate Y0 with
C              any reasonable accuracy. An error message is printed and
C              STRVH0 returns the value 0.0. The recommended value is
C                      XHIGH = 1/EPS.
C
C      For values of EPS and EPSNEG refer to the file MACHCON.TXT.
C
C      The machine-dependent constants are computed internally by
C      using the D1MACH subroutine.
C
C
C   INTRINSIC FUNCTIONS USED:
C
C      ABS, COS, SIN, SQRT.
C
C
C   OTHER MISCFUN SUBROUTINES USED:
C
C          CHEVAL , ERRPRN, D1MACH
C
C
C   AUTHOR:
C          ALLAN J. MACLEOD
C          DEPT. OF MATHEMATICS AND STATISTICS
C          UNIVERSITY OF PAISLEY
C          HIGH ST.
C          PAISLEY
C          SCOTLAND
C          PA1 2BE
C
C          (e-mail: macl_ms0@paisley.ac.uk )
C
C
C   LATEST REVISION:
C                   23 January, 1996
C
C
      INTEGER INDSGN,NTERM1,NTERM2,NTERM3,NTERM4
      DOUBLE PRECISION ARRH0(0:19),ARRH0A(0:20),AY0ASP(0:12),
     1     AY0ASQ(0:13),CHEVAL,EIGHT,ELEVEN,HALF,H0AS,
     2     ONEHUN,ONE,PIBY4,RT2BPI,SIXTP5,T,THR2P5,TWENTY,
     3     TWOBPI,TWO62,X,XHIGH,XLOW,XMP4,XSQ,XVALUE,
     4     Y0P,Y0Q,Y0VAL,ZERO,D1MACH
      CHARACTER FNNAME*6,ERRMSG*26*)
const
      FNNAME='STRVH0';
      ERRMSG='ARGUMENT TOO LARGE IN SIZE';
      ZERO:double=0.0;
      HALF:double=0.5;
      ONE: double=1.0;
      EIGHT: double=8.0;
      ELEVEN:double=11.0;
      TWENTY:double=20.0;
      ONEHUN:double=100.0;
      SIXTP5:double=60.5;
      TWO62: double=262.0;
      THR2P5: double= 302.5;
      PIBY4:double= 0.78539816339744830962;
      RT2BPI:double= 0.79788456080286535588;
      TWOBPI:double= 0.63661977236758134308;
      ARRH0: array[0..19] of double=(
       0.28696487399013225740E0,
      -0.25405332681618352305E0,
       0.20774026739323894439E0,
      -0.20364029560386585140E0,
       0.12888469086866186016E0,
      -0.4825632815622261202E-1,
       0.1168629347569001242E-1,
      -0.198118135642418416E-2,
       0.24899138512421286E-3,
      -0.2418827913785950E-4,
       0.187437547993431E-5,
      -0.11873346074362E-6,
       0.626984943346E-8,
      -0.28045546793E-9,
       0.1076941205E-10,
      -0.35904793E-12,
       0.1049447E-13,
      -0.27119E-15,
       0.624E-17,
      -0.13E-18
      );
      ARRH0A: array[0..20] of double=(
       1.99291885751992305515E0,
      -0.384232668701456887E-2,
      -0.32871993712353050E-3,
      -0.2941181203703409E-4,
      -0.267315351987066E-5,
      -0.24681031075013E-6,
      -0.2295014861143E-7,
      -0.215682231833E-8,
      -0.20303506483E-9,
      -0.1934575509E-10,
      -0.182773144E-11,
      -0.17768424E-12,
      -0.1643296E-13,
      -0.171569E-14,
      -0.13368E-15,
      -0.2077E-16,
       0.2E-19,
      -0.55E-18,
       0.10E-18,
      -0.4E-19,
       0.1E-19
      );
      AY0ASP: array[0..12] of double=(
       1.99944639402398271568E0,
      -0.28650778647031958E-3,
      -0.1005072797437620E-4,
      -0.35835941002463E-6,
      -0.1287965120531E-7,
      -0.46609486636E-9,
      -0.1693769454E-10,
      -0.61852269E-12,
      -0.2261841E-13,
      -0.83268E-15,
      -0.3042E-16,
      -0.115E-17,
      -0.4E-19
      );
      AY0ASQ: array[0..13] of double=(
       1.99542681386828604092E0,
      -0.236013192867514472E-2,
      -0.7601538908502966E-4,
      -0.256108871456343E-5,
      -0.8750292185106E-7,
      -0.304304212159E-8,
      -0.10621428314E-9,
      -0.377371479E-11,
      -0.13213687E-12,
      -0.488621E-14,
      -0.15809E-15,
      -0.762E-17,
      -0.3E-19,
      -0.3E-19
      );
{C
C   Start computation
C}
var
INDSGN,NTERM1,NTERM2,NTERM3,NTERM4: integer;
H0AS,T,X,XHIGH,XLOW,XMP4,XSQ,Y0P,Y0Q,Y0VAL, aux: double;

begin
      X := XVALUE;
      INDSGN := 1;
      IF ( X < ZERO ) THEN
         begin
         X := -X;
         INDSGN := -1;
         end;
{C
C   Compute the machine-dependent constants.
C}
      H0AS := D1MACH(3);
      XHIGH := ONE / D1MACH(4);
{C
C   Error test
C}
      IF ( ABS(XVALUE) > XHIGH ) THEN
         begin
         writeln(FNNAME,': ',ERRMSG);
         STRVH0 := ZERO;
         exit;
         end;
{C
C   continue with machine constants
C}
      T := H0AS / ONEHUN;
      IF ( X <= ELEVEN ) THEN
         begin
         {for NTERM1 := 19 downto 0 do
            IF ( ABS(ARRH0[NTERM1]) > T ) then break;}
         NTERM1 := 19;
            while (NTERM1>0) and (ARRH0[NTERM1]<=T) do
               NTERM1 := NTERM1-1;
         Y0P := SQRT ( H0AS );
         XLOW := Y0P + Y0P + Y0P;
         end
      ELSE
         begin
         {for NTERM2 := 20 downto 0 do
            IF ( ABS(ARRH0A[NTERM2]) > T ) then break;}
         NTERM2 := 20;
            while (NTERM2>0) and (ARRH0A[NTERM2]<=T) do
               NTERM2 := NTERM2-1;
         {for NTERM3 := 12 downto 0 do
            IF ( ABS(AY0ASP[NTERM3]) > T ) then break;}
         NTERM3 := 12;
            while (NTERM3>0) and (AY0ASP[NTERM3]<=T) do
               NTERM3 := NTERM3-1;
         {for NTERM4 := 13 downto 0 do
            IF ( ABS(AY0ASQ[NTERM4]) > T ) then break;}
         NTERM4 := 13;
            while (NTERM4>0) and (AY0ASQ[NTERM4]<=T) do
               NTERM4 := NTERM4-1;
         end;
{C
C   Code for abs(x) <= 11
C}
      IF ( X <= ELEVEN ) THEN
         IF ( X < XLOW ) THEN
            aux := TWOBPI * X
         ELSE
            begin
            T := ( ( X * X ) / SIXTP5 - HALF ) - HALF;
            aux := TWOBPI * X * CHEVAL ( NTERM1 , @ARRH0 , T );
            end
      ELSE
{C
C   Code for abs(x) > 11
C}
         begin
         XSQ := X * X;
         T := ( TWO62 - XSQ ) / ( TWENTY + XSQ );
         Y0P := CHEVAL ( NTERM3 , @AY0ASP , T );
         Y0Q := CHEVAL ( NTERM4 , @AY0ASQ , T ) / ( EIGHT * X );
         XMP4 := X - PIBY4;
         Y0VAL := Y0P * SIN ( XMP4 ) - Y0Q * COS ( XMP4 );
         Y0VAL := Y0VAL * RT2BPI / SQRT ( X );
         T := ( THR2P5 - XSQ ) / ( SIXTP5 + XSQ );
         H0AS := TWOBPI * CHEVAL ( NTERM2 , @ARRH0A , T ) / X;
         aux := Y0VAL + H0AS;
         end;
      IF ( INDSGN = -1 ) then STRVH0 := -aux else STRVH0:= aux;
END;


FUNCTION STRVH1(XVALUE: double): double;
(*C   DESCRIPTION:
C      This function calculates the value of the Struve function
C      of order 1, denoted H1(x), for the argument XVALUE, defined as
C
C                                                                  2
C        STRVH1(x) = (2x/pi) integral{0 to pi/2} sin( x cos(t))*sin t dt
C
C      H1 also satisfies the second-order differential equation
C
C                    2   2                   2            2
C                   x * D f  +  x * Df  +  (x - 1)f  =  2x / pi
C
C      The code uses Chebyshev expansions with the coefficients
C      given to 20D.
C
C
C   ERROR RETURNS:
C      As the asymptotic expansion of H1 involves the Bessel function
C      of the second kind Y1, there is a problem for large x, since
C      we cannot accurately calculate the value of Y1. An error message
C      is printed and STRVH1 returns the value 0.0.
C
C
C   MACHINE-DEPENDENT CONSTANTS:
C
C      NTERM1 - The no. of terms to be used in the array ARRH1. The
C               recommended value is such that
C                      ABS(ARRH1(NTERM1)) < EPS/100.
C
C      NTERM2 - The no. of terms to be used in the array ARRH1A. The
C               recommended value is such that
C                      ABS(ARRH1A(NTERM2)) < EPS/100.
C
C      NTERM3 - The no. of terms to be used in the array AY1ASP. The
C               recommended value is such that
C                      ABS(AY1ASP(NTERM3)) < EPS/100.
C
C      NTERM4 - The no. of terms to be used in the array AY1ASQ. The
C               recommended value is such that
C                      ABS(AY1ASQ(NTERM4)) < EPS/100.
C
C      XLOW1 - The value of x, below which H1(x) set to zero, if
C              abs(x)<XLOW1. The recommended value is
C                      XLOW1 = 1.5 * SQRT(XMIN)
C
C      XLOW2 - The value for which H1(x) = 2*x*x/pi to machine precision, if
C              abs(x) < XLOW2. The recommended value is
C                      XLOW2 = SQRT(15*EPSNEG)
C
C      XHIGH - The value above which we are unable to calculate Y1 with
C              any reasonable accuracy. An error message is printed and
C              STRVH1 returns the value 0.0. The recommended value is
C                      XHIGH = 1/EPS.
C
C      For values of EPS, EPSNEG and XMIN refer to the file MACHCON.TXT.
C
C      The machine-dependent constants are computed internally by
C      using the D1MACH subroutine.
C
C
C   INTRINSIC FUNCTIONS USED:
C
C      ABS, COS, SIN, SQRT.
C
C
C   OTHER MISCFUN SUBROUTINES USED:
C
C          CHEVAL , ERRPRN, D1MACH
C
C
C   AUTHOR:
C          ALLAN J. MACLEOD
C          DEPT. OF MATHEMATICS AND STATISTICS
C          UNIVERSITY OF PAISLEY
C          HIGH ST.
C          PAISLEY
C          SCOTLAND
C          PA1 2BE
C
C          (e-mail: macl_ms0@paisley.ac.uk)
C
C
C   LATEST REVISION:
C                   23 January, 1996
C
C *)
{      INTEGER NTERM1,NTERM2,NTERM3,NTERM4
      DOUBLE PRECISION ARRH1(0:17),ARRH1A(0:21),AY1ASP(0:14),
     1     AY1ASQ(0:15),CHEVAL,EIGHT,FIFTEN,FORTP5,HALF,
     2     H1AS,NINE,ONEHUN,ONE82,RT2BPI,T,THPBY4,
     3     TWENTY,TWOBPI,TW02P5,X,XHIGH,XLOW1,XLOW2,
     4     XM3P4,XSQ,XVALUE,Y1P,Y1Q,Y1VAL,ZERO,D1MACH
      CHARACTER FNNAME*6,ERRMSG*26}
var
nterm1, nterm2, nterm3, nterm4: integer;
h1as, t, x, xhigh, xlow1, xlow2, xm3p4, xsq, y1p, y1q, y1val: double;

{      DATA FNNAME/'STRVH1'/
      DATA ERRMSG/'ARGUMENT TOO LARGE IN SIZE'/
      DATA ZERO,HALF,EIGHT/0.0 D 0 , 0.5 D 0 , 8.0 D 0/
      DATA NINE,FIFTEN/ 9.0 D 0 , 15.0 D 0 /
      DATA TWENTY,ONEHUN/ 20.0 D 0 , 100.0 D 0/
      DATA FORTP5,ONE82,TW02P5/40.5 D 0 , 182.0 D 0 , 202.5 D 0/}
const
zero: double = 0.0;
half: double = 0.5;
eight: double = 8.0;
nine: double = 9.0;
fiften: double = 15.0;
twenty: double = 20.0;
onehun: double = 100.0;
fortp5: double = 40.5;
one82: double = 182.0;
tw02p5: double = 202.5;

RT2BPI: double = 0.79788456080286535588E0;
THPBY4: double = 2.35619449019234492885E0;
TWOBPI: double = 0.63661977236758134308E0;

ARRH1: array [0..17] of double = (
 0.17319061083675439319E0,
-0.12606917591352672005E0,
 0.7908576160495357500E-1,
-0.3196493222321870820E-1,
 0.808040581404918834E-2,
-0.136000820693074148E-2,
 0.16227148619889471E-3,
-0.1442352451485929E-4,
 0.99219525734072E-6,
-0.5441628049180E-7,
 0.243631662563E-8,
-0.9077071338E-10,
 0.285926585E-11,
-0.7716975E-13,
 0.180489E-14,
-0.3694E-16,
 0.67E-18,
-0.1E-19);

ARRH1A: array[0..21] of double = (
2.01083504951473379407E0,
0.592218610036099903E-2,
0.55274322698414130E-3,
0.5269873856311036E-4,
0.506374522140969E-5,
0.49028736420678E-6,
0.4763540023525E-7,
0.465258652283E-8,
0.45465166081E-9,
0.4472462193E-10,
0.437308292E-11,
0.43568368E-12,
0.4182190E-13,
0.441044E-14,
0.36391E-15,
0.5558E-16,
-0.4E-19,
0.163E-17,
-0.34E-18,
0.13E-18,
-0.4E-19,
0.1E-19);

AY1ASP: array[0..14] of double = (
2.00135240045889396402E0,
0.71104241596461938E-3,
0.3665977028232449E-4,
0.191301568657728E-5,
0.10046911389777E-6,
0.530401742538E-8,
0.28100886176E-9,
0.1493886051E-10,
0.79578420E-12,
0.4252363E-13,
0.227195E-14,
0.12216E-15,
0.650E-17,
0.36E-18,
0.2E-19);

AY1ASQ: array[0..15] of double = (
5.99065109477888189116E0,
-0.489593262336579635E-2,
-0.23238321307070626E-3,
-0.1144734723857679E-4,
-0.57169926189106E-6,
-0.2895516716917E-7,
-0.147513345636E-8,
-0.7596537378E-10,
-0.390658184E-11,
-0.20464654E-12,
-0.1042636E-13,
-0.57702E-15,
-0.2550E-16,
-0.210E-17,
0.2E-19,
-0.2E-19);

begin
{C
C   Start computation
C
      X = ABS ( XVALUE )}
x:= abs(xvalue);

{C
C   Compute the machine-dependent constants.
C
      XHIGH = ( HALF + HALF ) / D1MACH(4)}
xhigh:= (half+half)/d1mach(4);

{C
C   Error test
C
      IF ( X .GT. XHIGH ) THEN
         CALL ERRPRN(FNNAME,ERRMSG)
         STRVH1 = ZERO
         RETURN
      ENDIF}
if x>xhigh then
   begin
   writeln('StruveH1: x is too high');
   strvh1:= zero;
   exit;
   end;

{C
C   continue with machine constants
C
      H1AS = D1MACH(3)
      T = H1AS / ONEHUN}
h1as:= d1mach(3);
t:= h1as/onehun;

{      IF ( X .LE. NINE ) THEN
         DO 10 NTERM1 = 17 , 0 , -1
            IF ( ABS(ARRH1(NTERM1)) .GT. T ) GOTO 19
 10      CONTINUE
 19      XLOW1 = HALF * SQRT(D1MACH(1))
         XLOW1 = XLOW1 + XLOW1 + XLOW1
         XLOW2 = SQRT ( FIFTEN * H1AS )}
if x<= nine then
   begin
   nterm1:= 17;
   while (nterm1>0) and (abs(arrh1[nterm1])<=t) do dec(nterm1);
   xlow1:= half* sqrt(d1mach(1));
   xlow1:= xlow1 + xlow1 + xlow1;
   xlow2:= sqrt(fiften*h1as);
   end

{      ELSE
         DO 40 NTERM2 = 21 , 0 , -1
            IF ( ABS(ARRH1A(NTERM2)) .GT. T ) GOTO 49
 40      CONTINUE
 49      DO 50 NTERM3 = 14 , 0 , -1
            IF ( ABS(AY1ASP(NTERM3)) .GT. T ) GOTO 59
 50      CONTINUE
 59      DO 60 NTERM4 = 15 , 0 , -1
            IF ( ABS(AY1ASQ(NTERM4)) .GT. T ) GOTO 69
 60      CONTINUE
 69   ENDIF}
else
   begin
   nterm2:= 21;
   while (nterm2>0) and (abs(arrh1a[nterm2])<=t) do dec(nterm2);
   nterm3:= 14;
   while (nterm3>0) and (abs(ay1asp[nterm3])<=t) do dec(nterm3);
   nterm4:= 15;
   while (nterm4>0) and (abs(ay1asq[nterm4])<=t) do dec(nterm4);
   end;

{C
C   Code for abs(x) <= 9
C
      IF ( X .LE. NINE ) THEN
         IF ( X .LT. XLOW1 ) THEN
            STRVH1 = ZERO
         ELSE
            XSQ = X * X
            IF ( X .LT. XLOW2 ) THEN
               STRVH1 = TWOBPI * XSQ
            ELSE
               T = ( XSQ / FORTP5 - HALF ) - HALF
               STRVH1 = TWOBPI * XSQ * CHEVAL ( NTERM1 , ARRH1 , T )
            ENDIF
         ENDIF
      ELSE}

if x<= nine then
   if x< xlow1 then
      strvh1:= zero
   else
      begin
      xsq:= x*x;
      if x<xlow2 then
         strvh1:= twobpi*xsq
      else
         begin
         t:= (xsq/fortp5 - half) - half;
         strvh1:= twobpi*xsq*cheval(nterm1,@arrh1,t);
         end;
      end
else
{C
C   Code for abs(x) > 9
C
         XSQ = X * X
         T = ( ONE82 - XSQ ) / ( TWENTY + XSQ )
         Y1P = CHEVAL ( NTERM3 , AY1ASP , T )
         Y1Q = CHEVAL ( NTERM4 , AY1ASQ , T ) / ( EIGHT * X)
         XM3P4 = X - THPBY4
         Y1VAL = Y1P * SIN ( XM3P4 ) + Y1Q * COS ( XM3P4 )
         Y1VAL = Y1VAL * RT2BPI / SQRT ( X )
         T = ( TW02P5 - XSQ ) / ( FORTP5 + XSQ )
         H1AS = TWOBPI * CHEVAL ( NTERM2 , ARRH1A , T )
         STRVH1 = Y1VAL + H1AS
      ENDIF
      RETURN
      END}
   begin
   xsq:= x*x;
   t:= (one82-xsq)/(twenty+xsq);
   y1p:= cheval(nterm3,@ay1asp,t);
   y1q:= cheval(nterm4,@ay1asq,t)/(eight*x);
   xm3p4:= x-thpby4;
   y1val:= y1p*sin(xm3p4) + y1q*cos(xm3p4);
   y1val:= y1val*rt2bpi/sqrt(x);
   t:= (tw02p5-xsq)/(fortp5+xsq);
   h1as:= twobpi*cheval(nterm2,@arrh1a,t);
   strvh1:= y1val+h1as;
   end;
end;


{begin
firstinitds:= true;
firstdcsevl:= true;
firstd9b0mp:= true;
firstdbesj0:= true;

firstd9b1mp:= true;
firstdbesj1:= true;}


end.
