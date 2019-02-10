cmake_minimum_required(VERSION 3.10)
project(benchmark_cmatrix)

include(ExternalProject)
find_package(Git REQUIRED)

set(CMAKE_CXX_STANDARD 17)

# ------------------------------------------------------------------------------
# Add CMatrix
# ------------------------------------------------------------------------------
add_subdirectory(CMatrix)

# ------------------------------------------------------------------------------
# Add Celero as an external project
# ------------------------------------------------------------------------------

find_package(Git REQUIRED)
include(ExternalProject)

message(STATUS "Configuring Celero")
ExternalProject_Add(celero_project
        GIT_REPOSITORY https://github.com/DigitalInBlue/Celero.git
        GIT_TAG master
        GIT_SHALLOW 1
        UPDATE_COMMAND ""
        INSTALL_DIR ${CMAKE_BINARY_DIR}
        CMAKE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX:STRING=${CMAKE_BINARY_DIR}
        -DCELERO_COMPILE_DYNAMIC_LIBRARIES:BOOL=OFF
        -DCELERO_ENABLE_EXPERIMENTS:BOOL=OFF
        -DCELERO_ENABLE_FOLDERS:BOOL=OFF)
ExternalProject_Get_Property(celero_project SOURCE_DIR)
ExternalProject_Get_Property(celero_project INSTALL_DIR)
set(CELERO_INSTALL_DIR ${INSTALL_DIR})
message(STATUS "CELERO_INSTALL_DIR=${CELERO_INSTALL_DIR}")

include_directories(${CELERO_INSTALL_DIR}/include)
add_library(libcelero STATIC IMPORTED GLOBAL)
set_property(TARGET libcelero PROPERTY
        IMPORTED_LOCATION_DEBUG
        "${CELERO_INSTALL_DIR}/lib/static/${CMAKE_STATIC_LIBRARY_PREFIX}celerod${CMAKE_STATIC_LIBRARY_SUFFIX}")
set_property(TARGET libcelero PROPERTY
        IMPORTED_LOCATION
        "${CELERO_INSTALL_DIR}/lib/static/${CMAKE_STATIC_LIBRARY_PREFIX}celero${CMAKE_STATIC_LIBRARY_SUFFIX}")
add_dependencies(libcelero celero_project)

# ------------------------------------------------------------------------------
# Benchmark
# ------------------------------------------------------------------------------
include_directories(include)

set(HEADERS
        include/debug.hpp
        include/mkl_utils.hpp
        include/random_vector.hpp
        include/fixture.hpp)

# --- Print debug information
option(OPTION_PRINT_DEBUG_INFORMATION "Prints debug information" OFF)

add_executable(benchmark_dgemm ${HEADERS} benchmark_dgemm.cpp )

target_link_libraries(benchmark_dgemm cmatrix libcelero)

if (OPTION_PRINT_DEBUG_INFORMATION)
    add_compile_definitions(PRIVATE PRINT_DEBUG_INFORMATION)
    message(STATUS "Debug information will be printed to stdout")
endif ()
