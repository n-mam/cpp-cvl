project (cpp-cvl)

link_directories(
  C:/opencv/build/lib/Release
)

ADD_EXECUTABLE(
  counter
  Main.cpp 
)

SET_PROPERTY(TARGET counter PROPERTY CXX_STANDARD 17)

SET_TARGET_PROPERTIES(counter PROPERTIES 
  LINK_FLAGS "/INCREMENTAL:NO /DEBUG /OPT:REF /OPT:ICF"
)

TARGET_INCLUDE_DIRECTORIES(
 counter
 PRIVATE
 INCLUDE
 C:/opencv/build/install/include
)

TARGET_LINK_LIBRARIES(counter opencv_world440)