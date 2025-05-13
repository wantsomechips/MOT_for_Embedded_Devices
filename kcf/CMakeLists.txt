# cmake_minimum_required(VERSION 3.16)

# project(Embedded_MOT)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)


# find_package(OpenCV REQUIRED)

# include_directories(${OpenCV_INCLUDE_DIRS} ./)




add_library(kcf fhog.cpp kcftracker.cpp ffttools.cpp)