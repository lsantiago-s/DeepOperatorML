#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

gfortran -O2 -std=legacy -Wno-argument-mismatch \
  NUCLEOS.FOR \
  ZSOLVER.FOR \
  D1MACH.FOR \
  DQAGE.FOR \
  DQAGIE.FOR \
  DETERMINEkn.FOR \
  DETERMINEkn1.FOR \
  U1x.FOR \
  U1z.FOR \
  U2x.FOR \
  U2z.FOR \
  UZFx.FOR \
  UZMy.FOR \
  UZZ.FOR \
  DISPLACEMENTi.FOR \
  PRINCIPAL.FOR \
  -o multilayer_linux.exe
