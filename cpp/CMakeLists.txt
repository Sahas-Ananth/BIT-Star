cmake_minimum_required(VERSION 3.0)

project(BitStar)

set(CMAKE_CXX_STANDARD 11)
include_directories(include)
# install(DIRECTORY include/${PROJECT_NAME}/
#         DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
# add_executable(BitStar bitstar.cpp)
add_executable(BitStar unit_test2.cpp)