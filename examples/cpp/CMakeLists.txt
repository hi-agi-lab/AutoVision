cmake_minimum_required(VERSION 3.16)
project(autovision_cpp_reference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(postprocess_test src/postprocess_test.cpp)
target_include_directories(postprocess_test PRIVATE src)
set_target_properties(postprocess_test PROPERTIES CXX_EXTENSIONS OFF)

include(CTest)
if(BUILD_TESTING)
  add_test(NAME postprocess_test COMMAND postprocess_test)
endif()

# The host-side postprocess test builds with a bare C++ compiler. The
# ONNX Runtime example is optional: point ONNXRUNTIME_ROOT at an extracted
# onnxruntime release, and install OpenCV development files.
set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH
    "Extracted ONNX Runtime release directory")
find_package(OpenCV QUIET)
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS "${ONNXRUNTIME_ROOT}"
    PATH_SUFFIXES include include/onnxruntime/core/session)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
    HINTS "${ONNXRUNTIME_ROOT}"
    PATH_SUFFIXES lib lib64)

if(OpenCV_FOUND AND ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
  add_executable(autovision_infer src/main.cpp)
  target_include_directories(autovision_infer PRIVATE src
      "${ONNXRUNTIME_INCLUDE_DIR}")
  target_link_libraries(autovision_infer PRIVATE
      ${OpenCV_LIBS} "${ONNXRUNTIME_LIBRARY}")
else()
  message(STATUS
      "autovision_infer skipped: set ONNXRUNTIME_ROOT and install OpenCV "
      "(postprocess_test still builds)")
endif()
