cmake_minimum_required(VERSION 3.14)
project(bytetrack)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -MMD -Wall -Wextra -Winit-self")

# Check C++17 support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
if(COMPILER_SUPPORTS_CXX17)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
  add_definitions(-DCOMPILEDWITHC17)
  message(STATUS "Using flag -std=c++17.")
else()
  message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
endif()

find_package(Eigen3 3.3.0 QUIET)
if(NOT EIGEN3_FOUND)
  message(FATAL_ERROR "Eigen > 3.3.0 not found.")
endif()

find_package(OpenCV 3.4.0 QUIET)
if(NOT OpenCV_FOUND)
  message(FATAL_ERROR "OpenCV > 3.4.0 not found.")
endif()


set(RISCV OFF CACHE BOOL "The flag whether to use riscv or not")
if(RISCV)
  add_definitions(-DRISCV)
  add_definitions(-Wno-unused-result)
  message(STATUS "Use RISCV")
endif()

set(DPU OFF CACHE BOOL "The flag whether to use dpu or not")
if(DPU)
  message("Use DPU")
  add_definitions(-DDPU)
  # overwrite OoenCV_LIBS to avoid protobuf error
  set(OpenCV_LIBS "-lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui")
endif()

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIR}
  )

add_library(${PROJECT_NAME} SHARED
  ${PROJECT_SOURCE_DIR}/src/BYTETracker.cpp
  ${PROJECT_SOURCE_DIR}/src/KalmanFilter.cpp
  ${PROJECT_SOURCE_DIR}/src/FeatureProvider.cpp
  ${PROJECT_SOURCE_DIR}/src/LBPFeatureExtractor.cpp
  ${PROJECT_SOURCE_DIR}/src/lapjv.cpp
  ${PROJECT_SOURCE_DIR}/src/Object.cpp
  ${PROJECT_SOURCE_DIR}/src/Rect.cpp
  ${PROJECT_SOURCE_DIR}/src/STrack.cpp
  )

target_link_libraries(${PROJECT_NAME}
  ${EIGEN3_LIBS}
  ${OpenCV_LIBS}
  )
