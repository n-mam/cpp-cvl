# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file (GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file (GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

ie_add_lib(
  NAME fr
  SOURCES ${SOURCES}
  HEADERS ${HEADERS}
  INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include"
  DEPENDENCIES monitors
  OPENCV_DEPENDENCIES highgui)

if (WIN32)
  
  SET (CMAKE_CXX_FLAGS_RELEASE "/Zi /Od")

  SET_TARGET_PROPERTIES(
   fr PROPERTIES 
   LINK_FLAGS 
   "/DEBUG /OPT:REF /OPT:ICF"
  )

endif (WIN32)

target_link_libraries(fr PRIVATE ngraph::ngraph)
