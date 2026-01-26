cmake_minimum_required(VERSION 3.20)
project(kmeans_rvv
    VERSION 0.1
    DESCRIPTION "KMeans accuracy and performance tests for winter school (YADRO) (RVV)"
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_RVV "Enable RISC-V Vector intrinsics (RVV 1.0)" ON)

if(ENABLE_RVV)
    add_compile_definitions(ENABLE_RVV)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64gcv -mabi=lp64d")
    message(STATUS "RVV enabled")
else()
    message(STATUS "RVV disabled")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

find_package(OpenCV REQUIRED COMPONENTS core imgproc)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)
set(CMAKE_POLICY_DEFAULT_CMP0135 NEW) # i dont know about this line.....

FetchContent_MakeAvailable(googletest)

add_subdirectory(src)
add_subdirectory(tests)