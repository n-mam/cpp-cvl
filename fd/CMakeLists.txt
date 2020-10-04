# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
set(CMAKE_VERBOSE_MAKEFILE ON)

file (GLOB MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file (GLOB MAIN_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h*)

ovl_add_sample(NAME fd
               SOURCES ${MAIN_SRC}
               HEADERS ${MAIN_HEADERS}
               DEPENDENCIES monitors
               OPENCV_DEPENDENCIES highgui)

if (WIN32)
  
  SET (CMAKE_CXX_FLAGS_RELEASE "/Zi /Od")

  SET_TARGET_PROPERTIES(
   fd PROPERTIES 
   LINK_FLAGS 
   "/DEBUG /OPT:REF /OPT:ICF"
  )

endif (WIN32)

target_link_libraries(fd PRIVATE ngraph::ngraph)