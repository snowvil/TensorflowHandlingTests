cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(BasicTest LANGUAGES CXX)

find_package(tensorflow REQUIRED)
find_package(GTest REQUIRED)

macro(enhanceTarget targetname)
    target_sources(${targetname} PRIVATE testhelper.h)
    target_link_libraries(${targetname} PRIVATE tensorflow ${GTEST_BOTH_LIBRARIES})
    target_include_directories(${targetname} PRIVATE ${GTEST_INCLUDE_DIRS})
    target_compile_definitions(${targetname} PRIVATE COMPILER_MSVC)
    add_test(NAME ${targetname} COMMAND ${targetname})
endmacro()


add_executable(Const ConstTest.cpp)
enhanceTarget(Const)

add_executable(Variable VariableTest.cpp)
enhanceTarget(Variable)
