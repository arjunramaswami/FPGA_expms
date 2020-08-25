# Arjun Ramaswami
cmake_minimum_required(VERSION 3.10)

project(testkernels VERSION 0.1
            DESCRIPTION "Kernels for Experiments using FPGAs"
            LANGUAGES C CXX)

# OpenCL kernel targets generation
## setup cmake variables to generate header file
if(NOT DEFINED FPGA_BOARD_NAME)
  if(DEFINED $ENV{FPGA_BOARD_NAME})
    set(FPGA_BOARD_NAME $ENV{FPGA_BOARD_NAME} CACHE STRING "Target Board")
  else()
    set(FPGA_BOARD_NAME p520_hpc_sg280l CACHE STRING "Target Board")
  endif()
endif()

## Flags for different target options
set(AOC_FLAGS "-g -v -fp-relaxed -cl-single-precision-constant -no-interleaving=default" CACHE STRING "AOC compiler flags")
separate_arguments(AOC_FLAGS)
set(EMU_FLAGS "-legacy-emulator -march=emulator" CACHE STRING "AOC emulation flags")
separate_arguments(EMU_FLAGS)
set(REP_FLAGS "-report -rtl" CACHE STRING "AOC report flags")
separate_arguments(REP_FLAGS)
set(PROF_FLAGS "-profile=all" CACHE STRING "AOC profile flags")
separate_arguments(PROF_FLAGS)

if (INTELFPGAOPENCL_FOUND)
else()
  message(FATAL_ERROR, "Intel FPGA OpenCL SDK not found!")
endif()
