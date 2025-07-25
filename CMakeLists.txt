cmake_minimum_required(VERSION 3.16)
project(jinfer)
set(CMAKE_CXX_STANDARD 17)
find_package(benchmark REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(GTest REQUIRED)

set(link_lib glog::glog GTest::gtest)
if (!WIN32)
    set(link_lib "${link_lib} pthread")
endif ()
set(link_math_lib ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

aux_source_directory(./test/data DIR_TEST_DATA)
aux_source_directory(./test/runtime DIR_TEST_RUNTIME)

aux_source_directory(./source/data DIR_SOURCE_DATA)
aux_source_directory(./source/runtime DIR_SOURCE_RUNTIME)

add_executable(jinfer main.cpp ${DIR_TEST_DATA} ${DIR_TEST_RUNTIME} ${DIR_SOURCE_DATA} ${DIR_SOURCE_RUNTIME})
target_link_libraries(jinfer ${link_lib} ${link_math_lib} OpenMP::OpenMP_CXX)

target_include_directories(jinfer PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(jinfer PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(jinfer PUBLIC ${Armadillo_INCLUDE_DIR})
target_include_directories(jinfer PUBLIC ./include)

enable_testing()
