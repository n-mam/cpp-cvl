project (ocv-ppl-cnt)

find_package(Threads)

if (UNIX)
 find_package(OpenCV REQUIRED)
endif (UNIX)      

ADD_EXECUTABLE(counter Main.cpp)

SET_PROPERTY(TARGET counter PROPERTY CXX_STANDARD 17)

TARGET_INCLUDE_DIRECTORIES(
 counter
 PRIVATE
 INCLUDE
)

if (WIN32)
 TARGET_LINK_LIBRARIES(counter Ws2_32.lib)
else()
 TARGET_LINK_LIBRARIES(counter ${CMAKE_THREAD_LIBS_INIT} OpenCV)
endif (WIN32)