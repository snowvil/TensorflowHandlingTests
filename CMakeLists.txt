cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(BasicTest LANGUAGES CXX)

find_package(tensorflow REQUIRED)
find_package(GTest REQUIRED)

function(enhanceTarget targetname)
    target_sources(${targetname} PRIVATE testhelper.h)
    target_link_libraries(${targetname} PRIVATE tensorflow ${GTEST_BOTH_LIBRARIES})
    target_include_directories(${targetname} PRIVATE ${GTEST_INCLUDE_DIRS})
    target_compile_definitions(${targetname} PRIVATE COMPILER_MSVC)
    gtest_add_tests(TARGET ${targetname} 
		EXTRA_ARGS --gtest_output=xml:${targetname}Test)
endfunction()


add_executable(Const ConstTest.cpp)
enhanceTarget(Const)

add_executable(Variable VariableTest.cpp)
enhanceTarget(Variable)

add_executable(NNOps NNOpsTest.cpp)
enhanceTarget(NNOps)

add_executable(Training TrainingTest.cpp)
enhanceTarget(Training)

add_executable(ArrayOps ArrayOpsTest.cpp)
enhanceTarget(ArrayOps)

# for the graph_def usage, you might need the libprotobuf.lib which can be found at:
# <tensorflow-build-dir>/protobuf/src/protobuf/Release/libprotobuf.lib

set(protobuf-lib_DIR "" CACHE FILEPATH "Filepath to libprotobuf.lib. HINT: <tensorflow-build-dir>/protobuf/src/protobuf/Release/")
add_library(protobuf STATIC IMPORTED)
set_target_properties(protobuf PROPERTIES IMPORTED_LOCATION ${protobuf-lib_DIR})

add_executable(SaveAndRestore SaveAndRestoreTest.cpp)
target_link_libraries(SaveAndRestore PRIVATE protobuf)
enhanceTarget(SaveAndRestore)

