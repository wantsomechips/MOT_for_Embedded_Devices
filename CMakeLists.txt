cmake_minimum_required(VERSION 3.16)

project(Embedded_MOT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Static Library Output Directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
# Shared Library Output Directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
# Executable Library Output Directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ./ObjectDetect ./ObjectTrack ./src ./kcf)

add_subdirectory(./ObjectDetect)
add_subdirectory(./ObjectTrack)
add_subdirectory(./kcf)
add_subdirectory(./src)



