cmake_minimum_required(VERSION 3.10)
project(matrix_lib)

set(CMAKE_CXX_STANDARD 17)

# ------------------------------------------------------------------------------
# OpenMP
# ------------------------------------------------------------------------------
find_package(OpenMP REQUIRED)

# ------------------------------------------------------------------------------
# Intel MKL
# ------------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(MKL)
if (MKL_FOUND)
    message(STATUS "Search summary")
    message(STATUS "    MKL interface library        ${MKL_INTERFACE_LIBRARY}")
    message(STATUS "    MKL core library             ${MKL_CORE_LIBRARY}")
    message(STATUS "    MKL threading layer library  ${MKL_THREADING_LAYER_LIBRARY}")
    message(STATUS "    MKL include                  ${MKL_INCLUDE_DIRS}")
    message(STATUS "    MKL libraries                ${MKL_LIBRARIES}")
endif ()

# ------------------------------------------------------------------------------
# CMatrix
# ------------------------------------------------------------------------------
set(HEADERS
        include/CMatrix.h
        include/CTensor.h
        include/CVector.h
        include/resource.h
        include/stdafx.h
        include/targetver.h)

set(SOURCES
        src/CMatrix.cpp
        src/CTensor.cpp
        src/CVector.cpp
        src/dllmain.cpp
        src/stdafx.cpp)

add_library(cmatrix ${SOURCES} ${HEADERS})

target_include_directories(cmatrix PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${MKL_INCLUDE_DIRS})

target_link_libraries(cmatrix
        OpenMP::OpenMP_CXX
        ${MKL_LIBRARIES})

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------
add_subdirectory(tests)

# ------------------------------------------------------------------------------
# TODO: Benchmarks
# ------------------------------------------------------------------------------
