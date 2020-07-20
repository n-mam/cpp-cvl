project (cpp-cvl)

find_package(Threads)

if (UNIX)
 find_package(OpenCV REQUIRED)
endif (UNIX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")

link_directories(
  C:/opencv/build/lib/Debug
  C:/dlib/build/dlib/Debug
)

ADD_EXECUTABLE(
  counter
  Main.cpp 
  C:/dlib/dlib/all/source.cpp)

ADD_EXECUTABLE(
  mocap 
  Mocap.cpp 
  C:/dlib/dlib/all/source.cpp)

SET_PROPERTY(TARGET counter PROPERTY CXX_STANDARD 17)
SET_PROPERTY(TARGET mocap PROPERTY CXX_STANDARD 17)

TARGET_INCLUDE_DIRECTORIES(
 counter
 PRIVATE
 INCLUDE
 C:/dlib
 C:/opencv/build/install/include
)

TARGET_INCLUDE_DIRECTORIES(
 mocap
 PRIVATE
 INCLUDE
 C:/dlib
 C:/opencv/build/install/include
)

if (WIN32)
 TARGET_LINK_LIBRARIES(counter Ws2_32.lib opencv_world440d dlib19.20.99_debug_64bit_msvc1922)
 TARGET_LINK_LIBRARIES(mocap Ws2_32.lib opencv_world440d dlib19.20.99_debug_64bit_msvc1922) 
else()
 TARGET_LINK_LIBRARIES(counter ${CMAKE_THREAD_LIBS_INIT} OpenCV dlib::dlib)
 TARGET_LINK_LIBRARIES(counter ${CMAKE_THREAD_LIBS_INIT} OpenCV dlib::dlib) 
endif (WIN32)