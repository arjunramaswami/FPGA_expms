# Author: Arjun Ramaswami
cmake_minimum_required(VERSION 3.10)

## 
# Call function to create custom build commands
# Generates targets:
#   - ${kernel_name}_emu: to generate emulation binary
#   - ${kernel_name}_rep: to generate report
#   - ${kernel_name}_syn: to generate synthesis binary
##
set(CL_PATH "${testkernels_SOURCE_DIR}/empty")
set(kernels empty)

include(${testkernels_SOURCE_DIR}/cmake/genKernelTargets.cmake)

if (INTELFPGAOPENCL_FOUND)
  gen_fft_targets(${kernels})
endif()