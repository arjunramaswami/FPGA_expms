cmake_minimum_required(VERSION 3.11)

include(FetchContent)

## Argparse
FetchContent_Declare(
  argparse
  GIT_REPOSITORY      https://github.com/cofyc/argparse.git
  GIT_TAG             0d5f5d0745df14a3f373f7eed85bf524714f4524)

FetchContent_GetProperties(argparse)
if(NOT argparse_POPULATED)
  FetchContent_Populate(argparse)
  add_subdirectory(${argparse_SOURCE_DIR} ${argparse_BINARY_DIR})
endif()  

## Hlslib
FetchContent_Declare(
  hlslib
  GIT_REPOSITORY      https://github.com/definelicht/hlslib.git
  GIT_TAG             78124d8595baf88752fca7d2b1b7ae0e19377265)

FetchContent_GetProperties(hlslib)
if(NOT hlslib_POPULATED)
  FetchContent_Populate(hlslib)
  set(hlslib_SOURCE_DIR ${hlslib_SOURCE_DIR} PARENT_SCOPE)
endif()  