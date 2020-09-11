project (cpp-cvl)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
if (UNIX)
 find_package(OpenSSL REQUIRED)
endif (UNIX)

ADD_EXECUTABLE(Test Test.cpp)

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
)

if (WIN32)
  TARGET_LINK_LIBRARIES(Test ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})
else()
  TARGET_LINK_LIBRARIES(Test ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT} OpenSSL::SSL)
endif (WIN32)

