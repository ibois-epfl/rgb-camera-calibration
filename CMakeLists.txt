
cmake_minimum_required(VERSION 3.5)

project(TSlam VERSION "1.0.0" LANGUAGES CXX)
set(PROJECT_SOVERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON) #...is required...
set(CMAKE_CXX_EXTENSIONS ON) #...with compiler extensions like gnu++11
set(CMAKE_CXX_STANDARD 11) # C++11...

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV include dir: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
endif()

ADD_EXECUTABLE(calibrate src/calibrate.cpp)

target_include_directories(calibrate PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(calibrate ${OpenCV_LIBS})