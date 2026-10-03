#!/bin/bash
# build.sh: build the neo-air tools into tools/neo-air/bin against an existing llama.cpp build (default llama.cpp/build-metal)
#   BUILD=<llama.cpp build dir>  tools/neo-air/build.sh
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd); LC=$(cd "$D/../../llama.cpp" && pwd); B=${BUILD:-$LC/build-metal}
INC=(-I"$LC/include" -I"$LC/common" -I"$LC/ggml/include" -I"$LC/tools/split-prefill")
LIB=(-L"$B/bin" -lllama -lllama-common -lggml -lggml-base -Wl,-rpath,"$B/bin")
mkdir -p "$D/bin"
for t in sdref sdbench; do xcrun clang++ -std=c++17 -O2 "${INC[@]}" -o "$D/bin/$t" "$D/$t.cpp" "${LIB[@]}"; done
xcrun clang -O2 -fobjc-arc -o "$D/bin/gpuwarm" "$D/gpuwarm.m" -framework Metal -framework Foundation
xcrun clang -O2 -fobjc-arc -o "$D/bin/therm" "$D/therm.m" -framework Foundation
echo "built: $(ls "$D/bin" | tr '\n' ' ')"
