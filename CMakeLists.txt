project (cpp-cvl)

link_directories(
  C:/opencv/build/lib/Release
)

ADD_EXECUTABLE(
  Test
  Test.cpp 
)

SET_PROPERTY(TARGET Test PROPERTY CXX_STANDARD 17)

if (WIN32)
  SET (CMAKE_CXX_FLAGS "/Zi /Od")
  SET_TARGET_PROPERTIES(
   Test PROPERTIES 
   LINK_FLAGS 
   "/DEBUG /OPT:REF /OPT:ICF"
 )
endif (WIN32)

TARGET_INCLUDE_DIRECTORIES(
 Test
 PUBLIC
 ${CMAKE_SOURCE_DIR}
 ${CMAKE_SOURCE_DIR}/INCLUDE
 ${CMAKE_SOURCE_DIR}/../cpp-npl
 ${CMAKE_SOURCE_DIR}/../cpp-npl/INCLUDE
 C:/opencv/build/install/include
)

TARGET_LINK_LIBRARIES(Test opencv_world440)