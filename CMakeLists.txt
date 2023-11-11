cmake_minimum_required(VERSION 3.10)

project(HelloHPC)

# HDF5
find_package(HDF5 REQUIRED COMPONENTS C)
include_directories(${HDF5_INCLUDE_DIRS})

# Add executable for hellompi.cpp
add_executable(hellompi hellompi.cpp)

# Add executable for hellohdf.cpp
add_executable(hellohdf hellohdf.cpp)
target_link_libraries(hellohdf ${HDF5_C_LIBRARIES})
