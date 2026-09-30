#!/usr/bin/env bash
# Build the authors' BING (torrvision/Objectness, C++) and run it on a VOC-style folder.
#
#   baselines/original_bing/run_original.sh <dataset dir> [threads]
#
# <dataset dir> is a PASCAL VOC 2007 folder, or one produced by scripts/export_voc_format.py.
# The upstream code is cloned into third_party/ (not committed); only main.cpp is
# replaced by main_cli.cpp so the dataset path comes from the command line.
# Needs: git, cmake, a C++ compiler, OpenCV 4 dev files (apt install libopencv-dev).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SRC="$ROOT/third_party/Objectness"
DATA="$(cd "$1" && pwd)/"
THREADS="${2:-1}"

if [ ! -d "$SRC" ]; then
  git clone --depth 1 https://github.com/torrvision/Objectness "$SRC"
fi
cp "$HERE/main_cli.cpp" "$SRC/Src/main.cpp"
bash "$HERE/patch_opencv4.sh" "$SRC/Src"
mkdir -p "$SRC/Src/build"
(cd "$SRC/Src/build" && cmake -DCMAKE_BUILD_TYPE=Release .. >/dev/null && make -j"$(nproc)" >/dev/null)

rm -rf "$DATA/Results" "$DATA/Local"
echo "Running authors' BING on $DATA with $THREADS thread(s)"
OMP_NUM_THREADS="$THREADS" "$SRC/Src/build/BING_linux" "$DATA" | tee "$DATA/run.log"
