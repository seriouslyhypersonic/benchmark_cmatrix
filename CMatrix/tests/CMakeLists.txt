cmake_minimum_required(VERSION 3.10)

project(MatrixLibTest)

set(SOURCES test1.cpp)

add_executable(test1 ${SOURCES})

target_include_directories(test1 PRIVATE include)

target_link_libraries(test1 cmatrix)