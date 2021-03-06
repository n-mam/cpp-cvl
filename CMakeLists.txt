project (cpp-cvl)

set(CMAKE_VERBOSE_MAKEFILE ON)
# SET OPENCV_DIR=D:/opencv/build/install
# SET OPENSSL_ROOT_DIR=D:/vcpkg/installed/x64-windows

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(ngraph REQUIRED)
find_package(InferenceEngine 2.0 REQUIRED)

macro(ie_add_exe)
    set(options EXCLUDE_CPPLINT)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES HEADERS DEPENDENCIES OPENCV_DEPENDENCIES INCLUDE_DIRECTORIES)
    cmake_parse_arguments(IE_SAMPLE "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )

    # Find OpenCV components if exist
    if(IE_SAMPLE_OPENCV_DEPENDENCIES)
        find_package(OpenCV COMPONENTS ${IE_SAMPLE_OPENCV_DEPENDENCIES} QUIET)
        if(NOT OpenCV_FOUND)
            message(WARNING "OPENCV is disabled or not found, " ${IE_SAMPLE_NAME} " skipped")
            return()
        else()
            add_definitions(-DUSE_OPENCV)
        endif()
    endif()

    # Create named folders for the sources within the .vcproj
    # Empty name lists them directly under the .vcproj
    source_group("src" FILES ${IE_SAMPLE_SOURCES})
    if(IE_SAMPLE_HEADERS)
        source_group("include" FILES ${IE_SAMPLE_HEADERS})
    endif()

    # Create executable file from sources
    add_executable(${IE_SAMPLE_NAME} ${IE_SAMPLE_SOURCES} ${IE_SAMPLE_HEADERS})

    if(WIN32)
        set_target_properties(${IE_SAMPLE_NAME} PROPERTIES COMPILE_PDB_NAME ${IE_SAMPLE_NAME})
    endif()

    if(IE_SAMPLE_INCLUDE_DIRECTORIES)
        target_include_directories(${IE_SAMPLE_NAME} PRIVATE ${IE_SAMPLE_INCLUDE_DIRECTORIES})
    endif()
    target_include_directories(${IE_SAMPLE_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../common")

    target_link_libraries(${IE_SAMPLE_NAME} PRIVATE ${OpenCV_LIBRARIES} ${InferenceEngine_LIBRARIES}
                                                    ${IE_SAMPLE_DEPENDENCIES} )

    if(UNIX)
        target_link_libraries(${IE_SAMPLE_NAME} PRIVATE pthread)
    endif()

    # create global target with all samples / demo apps
    if(NOT TARGET ie_samples)
        add_custom_target(ie_samples ALL)
    endif()
    add_dependencies(ie_samples ${IE_SAMPLE_NAME})

    if(COMMAND add_cpplint_target AND NOT IE_SAMPLE_EXCLUDE_CPPLINT)
        add_cpplint_target(${IE_SAMPLE_NAME}_cpplint FOR_TARGETS ${IE_SAMPLE_NAME})
    endif()
endmacro()

 macro(ie_add_lib)
    set(options EXCLUDE_CPPLINT)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES HEADERS DEPENDENCIES OPENCV_DEPENDENCIES INCLUDE_DIRECTORIES)
    cmake_parse_arguments(IE_SAMPLE "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )

    # Find OpenCV components if exist
    if(IE_SAMPLE_OPENCV_DEPENDENCIES)
        find_package(OpenCV COMPONENTS ${IE_SAMPLE_OPENCV_DEPENDENCIES} QUIET)
        if(NOT OpenCV_FOUND)
            message(WARNING "OPENCV is disabled or not found, " ${IE_SAMPLE_NAME} " skipped")
            return()
        else()
            add_definitions(-DUSE_OPENCV)
        endif()
    endif()

    # Create named folders for the sources within the .vcproj
    # Empty name lists them directly under the .vcproj
    source_group("src" FILES ${IE_SAMPLE_SOURCES})
    if(IE_SAMPLE_HEADERS)
        source_group("include" FILES ${IE_SAMPLE_HEADERS})
    endif()

    # Create executable file from sources
    add_library(${IE_SAMPLE_NAME} STATIC ${IE_SAMPLE_SOURCES} ${IE_SAMPLE_HEADERS})

    if(WIN32)
        set_target_properties(${IE_SAMPLE_NAME} PROPERTIES COMPILE_PDB_NAME ${IE_SAMPLE_NAME})
    endif()

    if(IE_SAMPLE_INCLUDE_DIRECTORIES)
        target_include_directories(${IE_SAMPLE_NAME} PRIVATE ${IE_SAMPLE_INCLUDE_DIRECTORIES})
    endif()
    target_include_directories(${IE_SAMPLE_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../common")

    target_link_libraries(${IE_SAMPLE_NAME} PRIVATE ${OpenCV_LIBRARIES} ${InferenceEngine_LIBRARIES}
                                                    ${IE_SAMPLE_DEPENDENCIES} gflags)

    if(UNIX)
        target_link_libraries(${IE_SAMPLE_NAME} PRIVATE pthread)
    endif()

    # create global target with all samples / demo apps
    if(NOT TARGET ie_samples)
        add_custom_target(ie_samples ALL)
    endif()
    add_dependencies(ie_samples ${IE_SAMPLE_NAME})

    if(COMMAND add_cpplint_target AND NOT IE_SAMPLE_EXCLUDE_CPPLINT)
        add_cpplint_target(${IE_SAMPLE_NAME}_cpplint FOR_TARGETS ${IE_SAMPLE_NAME})
    endif()
 endmacro()

add_subdirectory(fr)

ie_add_exe(
    NAME TestCVL
    SOURCES Test.cpp)

ie_add_exe(
    NAME IETest
    SOURCES Test2.cpp)

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
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpp-osl
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpp-osl/INCLUDE
)

target_link_libraries(TestCVL PRIVATE fr)