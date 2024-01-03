cmake_minimum_required(VERSION 3.16)

project(coursepaper)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-O3)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

#add_executable(coursepaper Program_current3.cc)
#target_link_libraries(coursepaper ${CERES_LIBRARIES})

add_executable(myprog prog_copy.cpp)
target_link_libraries(myprog ${CERES_LIBRARIES})
