cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(OpsTest LANGUAGES CXX)

find_package(tensorflow REQUIRED)
find_package(GTest REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE tensorflow ${GTEST_BOTH_LIBRARIES})
target_include_directories(main PRIVATE ${GTEST_INCLUDE_DIRS})
target_compile_definitions(main PRIVATE COMPILER_MSVC)


add_test(NAME tester COMMAND main)