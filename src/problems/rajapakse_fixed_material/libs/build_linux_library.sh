#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build-fpc"
UNIT_DIR="${BUILD_DIR}/units"
BIN_DIR="${BUILD_DIR}/bin"
FIXED_OUT="${SCRIPT_DIR}/axsgrsce.so"
HOMO_OUT="${SCRIPT_DIR}/../../rajapakse_homogeneous/libs/axsgrsce.so"

mkdir -p "${UNIT_DIR}" "${BIN_DIR}"

compiler=""
if command -v fpc >/dev/null 2>&1; then
  compiler="fpc"
elif command -v ppcx64 >/dev/null 2>&1; then
  compiler="ppcx64"
fi

if [[ -z "${compiler}" ]]; then
  cat >&2 <<'EOF'
No Free Pascal compiler was found.

Install either `fpc` or `ppcx64`, then rerun this script. On Linux this script
builds a cluster-compatible `axsgrsce.so` from the Pascal sources in this
directory and copies it into both Rajapakse problem library folders.
EOF
  exit 1
fi

echo "Using compiler: ${compiler}"
echo "Building Rajapakse shared library from ${SCRIPT_DIR}/axsgrsce.pas"

rm -f "${BIN_DIR}/axsgrsce.so" "${BIN_DIR}/libaxsgrsce.so"

"${compiler}" \
  -B \
  -Mfpc \
  -O3 \
  -FU"${UNIT_DIR}" \
  -FE"${BIN_DIR}" \
  "${SCRIPT_DIR}/axsgrsce.pas"

built_lib=""
for candidate in \
  "${BIN_DIR}/axsgrsce.so" \
  "${BIN_DIR}/libaxsgrsce.so" \
  "${SCRIPT_DIR}/axsgrsce.so" \
  "${SCRIPT_DIR}/libaxsgrsce.so"
do
  if [[ -f "${candidate}" ]]; then
    built_lib="${candidate}"
    break
  fi
done

if [[ -z "${built_lib}" ]]; then
  echo "Build completed but no shared library was found in the expected locations." >&2
  exit 1
fi

cp "${built_lib}" "${FIXED_OUT}"
chmod 755 "${FIXED_OUT}"

mkdir -p "$(dirname "${HOMO_OUT}")"
cp "${built_lib}" "${HOMO_OUT}"
chmod 755 "${HOMO_OUT}"

echo "Built library: ${built_lib}"
echo "Installed:"
echo "  ${FIXED_OUT}"
echo "  ${HOMO_OUT}"
echo
echo "You can override the detected library path at runtime with:"
echo "  export RAJAPAKSE_AXSGRSCE_LIB=/absolute/path/to/axsgrsce.so"
