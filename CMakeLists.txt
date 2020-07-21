project (cpp-cvl)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")

link_directories(
  C:/opencv/build/lib/Debug
)

ADD_EXECUTABLE(
  counter
  Main.cpp 
)

ADD_EXECUTABLE(
  mocap 
  Mocap.cpp 
)

SET_PROPERTY(TARGET counter PROPERTY CXX_STANDARD 17)
SET_PROPERTY(TARGET mocap PROPERTY CXX_STANDARD 17)

TARGET_INCLUDE_DIRECTORIES(
 counter
 PRIVATE
 INCLUDE
 C:/opencv/build/install/include
)

TARGET_INCLUDE_DIRECTORIES(
 mocap
 PRIVATE
 INCLUDE
 C:/opencv/build/install/include
)

TARGET_LINK_LIBRARIES(counter opencv_world440d)
TARGET_LINK_LIBRARIES(mocap opencv_world440d) 