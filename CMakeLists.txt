project (cpp-cvl)

# set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenSSL REQUIRED)

# SET OPENCV_DIR=C:/opencv/build/install
# SET OPENSSL_ROOT_DIR=C:/vcpkg/installed/x64-windows

ADD_EXECUTABLE(TestCVL Test.cpp)

target_compile_definitions(TestCVL PUBLIC OPENVINO)

SET_PROPERTY(TARGET TestCVL PROPERTY CXX_STANDARD 17)

if (WIN32)
  
  SET (CMAKE_CXX_FLAGS_RELEASE "/Zi /Od")

  SET_TARGET_PROPERTIES(
   TestCVL PROPERTIES 
   LINK_FLAGS 
   "/DEBUG /OPT:REF /OPT:ICF"
  )

endif (WIN32)

TARGET_INCLUDE_DIRECTORIES(
  TestCVL
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/INCLUDE
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpp-npl
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpp-npl/INCLUDE
)

TARGET_LINK_LIBRARIES(
  TestCVL
  ${OpenCV_LIBS} 
  ${CMAKE_THREAD_LIBS_INIT} 
  OpenSSL::SSL
  fd
  pt
  fr
)
