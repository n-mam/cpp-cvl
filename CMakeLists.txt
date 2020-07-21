project (cpp-cvl)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")

link_directories(
  C:/opencv/build/lib/Release
)

ADD_EXECUTABLE(
  counter
  Main.cpp 
)

SET_PROPERTY(TARGET counter PROPERTY CXX_STANDARD 17)

TARGET_INCLUDE_DIRECTORIES(
 counter
 PRIVATE
 INCLUDE
 C:/opencv/build/install/include
)

TARGET_LINK_LIBRARIES(counter opencv_world440)