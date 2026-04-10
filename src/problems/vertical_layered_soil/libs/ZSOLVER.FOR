c
	SUBROUTINE ZSIST(AMAT,BMAT,IA,IB,XMAT)
c	
c	AMAT - Matriz do sistema linear;
c	BMAT - Vetor do sistema linear;
c	IA - Tamanho real da matriz;
c	IB - Tamanho com que a matriz foi declarada;
c	XMAT - Vetor com a resposta;
c	[AMAT]*{XMAT} = {BMAT}. 
c
c	Alteração em 15jan2008, J. Labaki.
c		1) Retirada a sentença PAUSE, dita obsoleta pelo compilador
c		   gfortran.
c	Alteração em 11mar2008, J. Labaki.
c		1) Inclusão do statement IMPLICIT NONE e declaração explícita
c		   dos inteiros I, IA e IB.
c	Comentário em 12mar2008, J. Labaki.
c		1) Cuidado! A matriz AMAT também é alterada ao passar por es-
c		   sa subrotina! O mesmo acontece com o vetor BMAT!
c	Alteração em 12mar2008, J. Labaki.
c		1) Declaração explícita da intenção IN das variáveis AMAT e
c		   BMAT, e intenção OUT da variável XMAT.
c	Observation in April 14th 2010, J. Labaki.
c		1) This solver do not assume that the input matrix
c		   is symmetric - it also works with non-symmetric
c		   linear systems.
c
	IMPLICIT NONE
c
	INTEGER INDX(2000000), I
	INTEGER, INTENT(IN) :: IA, IB
c
	COMPLEX*16, INTENT(IN) :: AMAT(IB,IB),BMAT(IB)
	COMPLEX*16, INTENT(OUT) :: XMAT(IB)
	COMPLEX*16 ZD
c
	CALL LUDCMZ(AMAT,IA,IB,INDX,ZD)
	CALL LUBKSZ(AMAT,IA,IB,INDX,BMAT)
c
	DO 10 I = 1,IA
		XMAT(I) = BMAT(I)
10	CONTINUE
c
	RETURN
c
	END SUBROUTINE ZSIST
c
c	***************************************************************
c
      SUBROUTINE LUDCMZ(ZA,N,NP,INDX,ZD)
c
c	Alteração em 11mar2008, J. Labaki.
c		1) Inclusão do statement IMPLICIT NONE e declaração explícita
c		   das seguintes variáveis:
c		   * os inteiros I, J, K, IMAX, NMAX, N e NP;
c		   * os complexos ZDUM2 (?), ZSUM e ZD;
c		   * os reais DUM, AAMAX e TINY.
c	
	IMPLICIT NONE
c
C	Variáveis definidas após adição de IMPLICIT NONE:
	INTEGER I, K, IMAX, J
	INTEGER, PARAMETER :: NMAX=600000
	INTEGER, INTENT(IN) :: N, NP
	COMPLEX*16 ZDUM2, ZSUM
	COMPLEX*16, INTENT(OUT) :: ZD
	REAL*8 DUM, AAMAX	
	REAL*8, PARAMETER :: TINY = 1.0D-20
c
c	Variáveis que já tinham sido definidas explicitamente:
	INTEGER INDX(2000000)
	COMPLEX*16 ZA(NP,NP),VV(NMAX)
c	
      ZD = DCMPLX(1.0D0, 0.0D0)
c
      DO 12 I=1,N
c
		AAMAX = 0.0D0
c
		DO 11 J=1,N
			IF (CDABS(ZA(I,J)).GT.AAMAX) THEN
				AAMAX=CDABS(ZA(I,J))
			END IF
11		CONTINUE
c
		IF (AAMAX.EQ.0.) GO TO 99
c
c		Antes de 15jan2008 era:
c		IF (AAMAX.EQ.0.) PAUSE 'Singular matrix.'
c
		VV(I)=1.D0/AAMAX
c
12    CONTINUE
c
      DO 19 J=1,N
c
		IF (J.GT.1) THEN
			DO 14 I=1,J-1
				ZSUM=ZA(I,J)
				IF (I.GT.1) THEN
					DO 13 K=1,I-1
						ZSUM=ZSUM-ZA(I,K)*ZA(K,J)
13					CONTINUE
					ZA(I,J)=ZSUM
				ENDIF
14			CONTINUE
		ENDIF
c
		AAMAX=0.0D0
c
		DO 16 I=J,N
			ZSUM=ZA(I,J)
			IF (J.GT.1)THEN
				DO 15 K=1,J-1
					ZSUM=ZSUM-ZA(I,K)*ZA(K,J)
15				CONTINUE
				ZA(I,J)=ZSUM
			ENDIF
			DUM=VV(I)*CDABS(ZSUM)
			IF (DUM.GE.AAMAX) THEN
				IMAX=I
				AAMAX=DUM
			ENDIF
16		CONTINUE
c
		IF (J.NE.IMAX)THEN
			DO 17 K=1,N
				ZDUM2=ZA(IMAX,K)
				ZA(IMAX,K)=ZA(J,K)
				ZA(J,K)=ZDUM2
17			CONTINUE
			ZD=-ZD
			VV(IMAX)=VV(J)
		ENDIF
c
		INDX(J)=IMAX
c
		IF(J.NE.N)THEN
			IF(CDABS(ZA(J,J)).EQ.0.) THEN
				ZA(J,J) = DCMPLX(TINY,TINY)
			END IF
			ZDUM2=1.D0/ZA(J,J)
			DO 18 I=J+1,N
				ZA(I,J) = ZA(I,J)*ZDUM2
18			CONTINUE
		ENDIF
c
19    CONTINUE
c
      IF(CDABS(ZA(N,N)).EQ.0.) THEN
		ZA(N,N) = DCMPLX(TINY,TINY)
	END IF
c
99    RETURN
c
      END SUBROUTINE LUDCMZ
c
c	***************************************************************
c
      SUBROUTINE LUBKSZ(ZA,N,NP,INDX,ZB)
c
c	Alteração em 11mar2008, J. Labaki.
c		1) Inclusão do statement IMPLICIT NONE e declaração explícita
c		   das seguintes variáveis:
c		   * os inteiros I, J, K, IMAX, NMAX, N e NP;
c		   * os complexos ZDUM2 (?), ZSUM e ZD;
c		   * os reais DUM, AAMAX e TINY.
c
	IMPLICIT NONE
c
C	Variáveis definidas após adição de IMPLICIT NONE:
	INTEGER I, J, II
	INTEGER, INTENT(IN) :: N, NP
	INTEGER, PARAMETER :: NMAX=600000
	REAL*8, PARAMETER :: TINY=1.0D-20
	INTEGER LL
	COMPLEX*16 ZSUM
c
C	Variáveis que já tinham sido definidas explicitamente:	
	COMPLEX*16 ZA(NP,NP),ZB(NP)
	INTEGER INDX(2000000)
c
      II = 0
c
      DO 12 I=1,N
c
		LL=INDX(I)
		ZSUM=ZB(LL)
		ZB(LL)=ZB(I)
c
		IF (II.NE.0) THEN
			DO 11 J=II,I-1
				ZSUM=ZSUM-ZA(I,J)*ZB(J)
11			CONTINUE
		ELSE IF (CDABS(ZSUM).NE.0.) THEN
			II=I
		ENDIF
c
		ZB(I)=ZSUM
c
12    CONTINUE
c
      DO 14 I=N,1,-1
c
		ZSUM=ZB(I)
c
		IF(I.LT.N)THEN
			DO 13 J=I+1,N
				ZSUM=ZSUM-ZA(I,J)*ZB(J)
13			CONTINUE
		ENDIF
c
		ZB(I)=ZSUM/ZA(I,I)
c
14	CONTINUE
c
      RETURN
c
      END SUBROUTINE LUBKSZ