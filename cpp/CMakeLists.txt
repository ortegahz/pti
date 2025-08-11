cmake_minimum_required(VERSION 3.10)
project(signal_demo LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ncnn REQUIRED)

find_package(OpenCV REQUIRED)

# --------------------- 线程库 ---------------------
find_package(Threads REQUIRED)

# --------------------- FFTW3 (double) -------------
find_package(PkgConfig REQUIRED)
# 优先用 pkg-config
pkg_check_modules(FFTW3_PKG QUIET fftw3)
if (FFTW3_PKG_FOUND)
    message(STATUS "FFTW3 found via pkg-config")
    set(FFTW3_INCLUDE_DIRS ${FFTW3_PKG_INCLUDE_DIRS})
    set(FFTW3_LIBRARIES ${FFTW3_PKG_LIBRARIES})
else ()
    # 手动搜索
    find_path(FFTW3_INCLUDE_DIR fftw3.h)
    find_library(FFTW3_LIBRARY NAMES fftw3)
    if (NOT FFTW3_INCLUDE_DIR OR NOT FFTW3_LIBRARY)
        message(FATAL_ERROR "Could not find fftw3. Please install fftw3-dev")
    endif ()
    set(FFTW3_INCLUDE_DIRS ${FFTW3_INCLUDE_DIR})
    set(FFTW3_LIBRARIES ${FFTW3_LIBRARY})
endif ()

# --------------------- ONNX Runtime ---------------
# 你之前把 ORT 解压到 /opt/onnxruntime
# 也可以通过 -DONNXRUNTIME_ROOT=... 在 cmake 命令行传入
if (NOT DEFINED ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_ROOT /opt/onnxruntime)
endif ()

find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
        PATHS ${ONNXRUNTIME_ROOT}/include)

find_library(ORT_LIBRARY onnxruntime
        PATHS ${ONNXRUNTIME_ROOT}/lib)

if (NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
    message(FATAL_ERROR "Cannot find onnxruntime in ${ONNXRUNTIME_ROOT}")
endif ()

# --------------------------------------------------
# 可执行文件 1：signal_processor
# --------------------------------------------------
add_executable(signal_processor
        signal_processor.cpp)

target_include_directories(signal_processor PRIVATE
        ${ORT_INCLUDE_DIR}
        ${FFTW3_INCLUDE_DIRS})

target_link_libraries(signal_processor PRIVATE
        ${ORT_LIBRARY}
        ${FFTW3_LIBRARIES}
        Threads::Threads)

# --------------------------------------------------
# 如果你还想编译 demo_onnx.cpp
# --------------------------------------------------
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/demo_onnx.cpp)
    add_executable(demo_onnx demo_onnx.cpp)
    target_include_directories(demo_onnx PRIVATE
            ${ORT_INCLUDE_DIR}
            ${FFTW3_INCLUDE_DIRS})
    target_link_libraries(demo_onnx PRIVATE
            ${ORT_LIBRARY}
            ${FFTW3_LIBRARIES}
            Threads::Threads)
endif ()

# --------------------------------------------------
# demo_ncnn.cpp
# --------------------------------------------------
add_executable(demo_ncnn demo_ncnn.cpp)
if (TARGET ncnn::ncnn)
    target_link_libraries(demo_ncnn PRIVATE ncnn::ncnn)
else ()
    target_link_libraries(demo_ncnn PRIVATE ncnn)
endif ()

# --------------------------------------------------
# signal_processor_ncnn.cpp
# --------------------------------------------------
add_executable(signal_processor_ncnn signal_processor_ncnn.cpp)
if (TARGET ncnn::ncnn)
    target_link_libraries(signal_processor_ncnn
            PRIVATE
            ncnn::ncnn
            ${FFTW3_LIBRARIES})
else ()
    target_link_libraries(signal_processor_ncnn
            PRIVATE
            ncnn
            ${FFTW3_LIBRARIES})
endif ()

# =============================================================
# signal_processor_opencv.cpp
# =============================================================
add_executable(signal_processor_opencv signal_processor_opencv.cpp)

target_include_directories(signal_processor_opencv PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${FFTW3_INCLUDE_DIRS})

target_link_libraries(signal_processor_opencv PRIVATE
        ${OpenCV_LIBS}
        ${FFTW3_LIBRARIES}
        Threads::Threads)

message(STATUS "==========================================")
message(STATUS "  FFTW3 include : ${FFTW3_INCLUDE_DIRS}")
message(STATUS "  FFTW3 libs    : ${FFTW3_LIBRARIES}")
message(STATUS "  ORT  include  : ${ORT_INCLUDE_DIR}")
message(STATUS "  ORT  library  : ${ORT_LIBRARY}")
message(STATUS "==========================================")