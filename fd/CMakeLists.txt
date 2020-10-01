# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file (GLOB MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file (GLOB MAIN_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h*)

ovl_add_sample(NAME fd
               SOURCES ${MAIN_SRC}
               HEADERS ${MAIN_HEADERS}
               DEPENDENCIES monitors
               OPENCV_DEPENDENCIES highgui)

target_link_libraries(fd PRIVATE ngraph::ngraph)