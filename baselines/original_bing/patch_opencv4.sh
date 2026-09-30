#!/usr/bin/env bash
# The upstream code (2014-2017) targets OpenCV 2/3 and uses legacy C constants
# (CV_BGR2GRAY, CV_INTER_NN, CV_StsAssert, ...). OpenCV 4 still ships them in the
# *_c.h compatibility headers, so including those is the only change needed.
set -euo pipefail
HDR="$1/kyheader.h"
if ! grep -q "imgproc_c.h" "$HDR"; then
  sed -i 's|#include <opencv2/opencv.hpp>|#include <opencv2/opencv.hpp>\n#include <opencv2/core/core_c.h>\n#include <opencv2/imgproc/imgproc_c.h>\n#include <opencv2/imgcodecs/legacy/constants_c.h>|' "$HDR"
fi

# OpenCV 4 has no cv::DataType<long>, so Mat_<INT64> no longer compiles. The code only
# ever accesses these buffers through ptr<INT64>(), so allocate them as untyped 8-byte
# matrices (zero-filled, same memory layout); the bitwise algorithm is unchanged.
TIG="$1/FilterTIG.cpp"
sed -i 's|Mat_<INT64> Tig1 = Mat_<INT64>::zeros(sz), Tig2 = Mat_<INT64>::zeros(sz);|Mat Tig1 = Mat::zeros(sz, CV_64F), Tig2 = Mat::zeros(sz, CV_64F);|; s|Mat_<INT64> Tig4 = Mat_<INT64>::zeros(sz), Tig8 = Mat_<INT64>::zeros(sz);|Mat Tig4 = Mat::zeros(sz, CV_64F), Tig8 = Mat::zeros(sz, CV_64F);|' "$TIG"

# Link order: the static BING library must come before the OpenCV libraries it uses,
# otherwise modern linkers (--as-needed) drop highgui and imshow is unresolved.
sed -i 's|target_link_libraries(${PROJECT_NAME} opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs ${EXTERNAL_LIBS} BING LibLinear)|target_link_libraries(${PROJECT_NAME} BING LibLinear ${OpenCV_LIBS} ${EXTERNAL_LIBS})|' "$1/CMakeLists.txt"
