
add_library(funcs funcs.cpp)

add_executable(main main.cpp)

target_link_libraries(main funcs ${OpenCV_LIBS} objDetect kcf)