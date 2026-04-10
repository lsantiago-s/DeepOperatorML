C	
	MODULE NUCLEOS
C
	CONTAINS
C
C	********** DISPLACEMENT FUNCTIONS ************
C
	REAL*8 FUNCTION RURFxint(ZETA)
C
	REAL*8, INTENT(IN) :: ZETA
	REAL*8 :: RU1x, RU2x
C
C	Returns the Real parcel of URFx
	CALL U1x(ZETA, 0, RU1x)
	CALL U2x(ZETA, 0, RU2x)
C
	RURFxint = (RU1x + RU2x)/2.0D0
C
	RETURN
	END FUNCTION RURFxint
C
C	**********************************************
C
	REAL*8 FUNCTION IURFxint(ZETA)
C
	REAL*8, INTENT(IN) :: ZETA
	REAL*8 :: IU1x, IU2x
C
C	Returns the Imaginary parcel of URFx
	CALL U1x(ZETA, 1, IU1x)
	CALL U2x(ZETA, 1, IU2x)
C
	IURFxint = (IU1x + IU2x)/2.0D0
C
	RETURN
	END FUNCTION IURFxint
C
C	**********************************************
C	**********************************************
C
	REAL*8 FUNCTION RURMyint(ZETA)
C
	REAL*8, INTENT(IN) :: ZETA
	REAL*8 :: RU1z, RU2z
C
C	Returns the Real parcel of URMy
	CALL U1z(ZETA, 0, RU1z)
	CALL U2z(ZETA, 0, RU2z)
C
	RURMyint = (RU1z + RU2z)/2.0D0
C
	RETURN
	END FUNCTION RURMyint
C
C	**********************************************
C
	REAL*8 FUNCTION IURMyint(ZETA)
C
	REAL*8, INTENT(IN) :: ZETA
	REAL*8 :: IU1z, IU2z
C
C	Returns the Imaginary parcel of URMy
	CALL U1z(ZETA, 1, IU1z)
	CALL U2z(ZETA, 1, IU2z)
C
	IURMyint = (IU1z + IU2z)/2.0D0
C
	RETURN
	END FUNCTION IURMyint
C
C	**********************************************
C	**********************************************
C
	REAL*8 FUNCTION RUZFxint(ZETA)
C
	REAL*8, INTENT(IN) :: ZETA
C
C	Returns the Real parcel of UZFx
	CALL UZFx(ZETA, 0, RUZFxint)
C
	RETURN
	END FUNCTION RUZFxint
C
C	**********************************************
C
	REAL*8 FUNCTION IUZFxint(ZETA)
C
	REAL*8, INTENT(IN) :: ZETA
C
C	Returns the Imaginary parcel of UZFx
	CALL UZFx(ZETA, 1, IUZFxint)
C
	RETURN
	END FUNCTION IUZFxint
C
C	**********************************************
C	**********************************************
C
	REAL*8 FUNCTION RUZMyint(ZETA)
C
	REAL*8, INTENT(IN) :: ZETA
C
C	Returns the Real parcel of UZMy
	CALL UZMy(ZETA, 0, RUZMyint)
C
	RETURN
	END FUNCTION RUZMyint
C
C	**********************************************
C
	REAL*8 FUNCTION IUZMyint(ZETA)
C
	REAL*8, INTENT(IN) :: ZETA
C
C	Returns the Imaginary parcel of UZMy
	CALL UZMy(ZETA, 1, IUZMyint)
C
	RETURN
	END FUNCTION IUZMyint
C
C	**********************************************
C	**********************************************
C
	REAL*8 FUNCTION RUZZint(ZETA)
C
	REAL*8, INTENT(IN) :: ZETA
C
C	Returns the Real parcel of UZZ (vertical displacement due to vertical load)
	CALL UZZ(ZETA, 0, RUZZint)
C
	RETURN
	END FUNCTION RUZZint
C
C	**********************************************
C
	REAL*8 FUNCTION IUZZint(ZETA)
C
	REAL*8, INTENT(IN) :: ZETA
C
C	Returns the Imaginary parcel of UZZ (vertical displacement due to vertical load)
	CALL UZZ(ZETA, 1, IUZZint)
C
	RETURN
	END FUNCTION IUZZint
C
C	**********************************************
C
C	*************** OTHER  FUNCTIONS *************
C
CC	**********************************************
CC
C	FUNCTION cH0(S1, S2, KSI)
CC
C	REAL*8, INTENT(IN) :: S1, S2
C	COMPLEX*16, INTENT(IN) :: KSI
C	COMPLEX*16 :: cH0, cJ1S1, cJ1S2
CC
C	CALL DCBJS(1.0D0, KSI*S2, 1, cJ1S1)
C	CALL DCBJS(1.0D0, KSI*S2, 1, cJ1S2)
C	cH0 = S2*cJ1S2 - S1*cJ1S1
C	cH0 = cH0/KSI
CC
C	RETURN
C	END FUNCTION cH0
CC
CC	**********************************************
C
	FUNCTION H0(S1, S2, KSI)
C
	REAL*8, INTENT(IN) :: S1, S2, KSI
	REAL*8 :: H0
C
	H0 = S2*BESSEL_JN(1, KSI*S2)
     *     - S1*BESSEL_JN(1, KSI*S1)
	H0 = H0/KSI
C
	RETURN
	END FUNCTION H0
C
CC	**********************************************
CC
CC
C	FUNCTION cH1(S1, S2, KSI)
CC
C	REAL*8, INTENT(IN) :: S1, S2
C	COMPLEX*16, INTENT(IN) :: KSI
C	COMPLEX*16 :: cH1, CJ2, CJ1
C	REAL*8 :: PI
CC	
C	PI = 3.141592653589793238462643383279D0
CC
C	CALL DCBJS(2.0D0, KSI*S2, 1, CJ2)
C	IF (S1.EQ.0.0D0) THEN
C		CJ1=0.0D0
C	ELSE
C		CALL DCBJS(2.0D0, KSI*S1, 1, CJ1)
C	END IF
C	cH1 = S2**2.0D0*CJ2 - S1**2.0D0*CJ1
C	cH1 = cH1/KSI
CC
C	RETURN
C	END FUNCTION cH1
CC
CC	**********************************************
C
	FUNCTION H1(S1, S2, KSI)
C
	REAL*8, INTENT(IN) :: S1, S2, KSI
	REAL*8 :: H1, PI
C	
	PI = 3.141592653589793238462643383279D0
C
	H1 = S2**2.0D0*BESSEL_JN(2, KSI*S2)
     *     - S1**2.0D0*BESSEL_JN(2, KSI*S1)
	H1 = H1/KSI
C
	RETURN
	END FUNCTION H1
C
C	**********************************************
C
	FUNCTION PS(S1, S2, KSI)
C
	REAL*8, INTENT(IN) :: S1, S2, KSI
	REAL*8 :: PS
C
	PS = S2*BESSEL_JN(1, KSI*S2)
     *     - S1*BESSEL_JN(1, KSI*S1)
	PS = PS/KSI
C
	RETURN
	END FUNCTION PS
C
C	**********************************************
C
	FUNCTION Jm(M, LAMBDA, R)
C
	INTEGER, INTENT(IN) :: M
	REAL*8, INTENT(IN) :: LAMBDA
	REAL*8, INTENT(IN) :: R
	REAL*8 :: Jm
C
	Jm = BESSEL_JN(M, LAMBDA*R)
C
	RETURN
	END FUNCTION Jm

C
C	**********************************************
C
	END MODULE NUCLEOS
C
