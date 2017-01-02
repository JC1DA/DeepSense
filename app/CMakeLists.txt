# Sets the minimum version of CMake required to build the native
# library. You should either keep the default value or only pass a
# value of 3.4.0 or lower.

cmake_minimum_required(VERSION 3.4.1)

# configure import libs
set(distribution_DIR ${CMAKE_SOURCE_DIR}/../distribution)
set(source_DIR ${CMAKE_SOURCE_DIR}/src/main/cpp)

add_library(lib_opencl SHARED IMPORTED)
set_target_properties( lib_opencl PROPERTIES IMPORTED_LOCATION ${distribution_DIR}/opencl/lib/${ANDROID_ABI}/libGLES_mali.so )
#set_target_properties( lib_opencl PROPERTIES IMPORTED_LOCATION ${distribution_DIR}/opencl/lib/${ANDROID_ABI}/libllvm-qcom.so )
#set_target_properties( lib_opencl PROPERTIES IMPORTED_LOCATION ${distribution_DIR}/opencl/lib/${ANDROID_ABI}/libOpenCL.so )

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds it for you.
# Gradle automatically packages shared libraries with your APK.

# build application's shared lib
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}    -Xlinker --no-warn-mismatch -O2 -mfpu=vfpv3-d16 -mhard-float -D_NDK_MATH_NO_SOFTFP=1 -march=armv7-a -mfloat-abi=hard")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Xlinker --no-warn-mismatch -O2 -mfpu=vfpv3-d16 -mhard-float -D_NDK_MATH_NO_SOFTFP=1 -march=armv7-a -mfloat-abi=hard -std=gnu++11")
set(STLPORT_FORCE_REBUILD "true")

add_library( # Sets the name of the library.
             deepsense

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             # Associated headers in the same location as their source
             # file are automatically included.
             src/main/cpp/deepsense.cpp
             ${source_DIR}/deepsense_lib.cpp
             ${source_DIR}/clio.cpp
             ${source_DIR}/basic_functions.cpp
             ${source_DIR}/utilities.cpp
             ${source_DIR}/classifier.cpp
             ${source_DIR}/layers/conv_layer.cpp
             ${source_DIR}/layers/maxpool.cpp
             ${source_DIR}/layers/fully_connected.cpp
             ${source_DIR}/layers/softmax.cpp
             ${source_DIR}/layers/lrn.cpp)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because system libraries are included in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

target_include_directories(deepsense PRIVATE ${distribution_DIR}/opencl/include ${distribution_DIR}/clblast/include)
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/include)

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log
              m_hard)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in the
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
                       deepsense

                       lib_opencl

                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib} )
