cmake_minimum_required(VERSION 3.14)
project(generate_submit_file)

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

set(BYTETRACK_DIR "${PROJECT_SOURCE_DIR}/../../")

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${BYTETRACK_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIR}
  )

set(RISCV OFF CACHE BOOL "The flag whether to use riscv or not")
if(RISCV)
  message("Use RISCV")
  add_definitions(-DRISCV)
endif()

set(DPU OFF CACHE BOOL "The flag whether to use dpu or not")
if(DPU)
  message("Use DPU")
  add_definitions(-DDPU)
  set(OpenCV_LIBS "-lopencv_core -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui")
  set(DPU_LIBS "-lglog -lvitis_ai_library-xnnpp -lvitis_ai_library-model_config -lprotobuf -lvitis_ai_library-dpu_task")
endif()

link_directories(
    ${BYTETRACK_DIR}/build
    )

add_executable(${PROJECT_NAME}
  ${PROJECT_SOURCE_DIR}/main.cpp
  ${PROJECT_SOURCE_DIR}/json11/json11.cpp
  )

target_link_libraries(${PROJECT_NAME}
  bytetrack
  ${EIGEN3_LIBS}
  ${OpenCV_LIBS}
  ${DPU_LIBS}
  )
