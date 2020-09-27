project (cpp-cvl)

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenSSL REQUIRED)

# SET OPENCV_DIR=C:/opencv/build/install
# SET OPENSSL_ROOT_DIR=C:/vcpkg/installed/x64-windows

ADD_EXECUTABLE(Test Test.cpp)

SET_PROPERTY(TARGET Test PROPERTY CXX_STANDARD 17)

if (WIN32)
  
  SET (CMAKE_CXX_FLAGS_RELEASE "/Zi /Od")

  SET_TARGET_PROPERTIES(
   Test PROPERTIES 
   LINK_FLAGS 
   "/DEBUG /OPT:REF /OPT:ICF"
  )

endif (WIN32)

TARGET_INCLUDE_DIRECTORIES(
  Test
  PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/INCLUDE
  ${CMAKE_SOURCE_DIR}/../cpp-npl
  ${CMAKE_SOURCE_DIR}/../cpp-npl/INCLUDE
)

TARGET_LINK_LIBRARIES(Test ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT} OpenSSL::SSL)
