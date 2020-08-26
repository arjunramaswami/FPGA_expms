// Author: Arjun Ramaswami

/**
 * @file bare.h
 * @brief Header file that provides APIs for OpenCL Host code
 */

#ifndef BARE_H
#define BARE_H

#include<stdbool.h>
/**
 * Single Precision Complex Floating Point Data Structure
 */
typedef struct {
  float x; /**< real value */
  float y; /**< imaginary value */
} float2;

/**
 * Double Precision Complex Floating Point Data Structure
 */
typedef struct {
  double x; /**< real value */
  double y; /**< imaginary value */
} double2;

/**
 * Record time in milliseconds of different FPGA runtime stages
 */
typedef struct fpga_timing {
  double pcie_read_t;   /**< Time to read from DDR to host using PCIe bus */ 
  double pcie_write_t; /**< Time to write from DDR to host using PCIe bus */ 
  double exec_t;      /**< Kernel execution time */
  int valid;          /**< Represents 1 signifying valid execution */
} fpga_t;

/** 
 * @brief Initialize FPGA
 * @param platform_name: name of the OpenCL platform
 * @param path         : path to binary
 * @param use_svm      : 1 if true 0 otherwise
 * @return 0 if successful 
          -1 Path to binary missing
          -2 Unable to find platform passed as argument
          -3 Unable to find devices for given OpenCL platform
          -4 Failed to create program, file not found in path
          -5 Device does not support required SVM
 */
extern int fpga_initialize(const char *platform_name, const char *path, int use_svm);

/** 
 * @brief Release FPGA Resources
 */
extern void fpga_final();

/** 
 * @brief Allocate memory of double precision complex floating points
 * @param sz  : size_t - size to allocate
 * @return void ptr or NULL
 */
extern void* fpga_complex_malloc(size_t sz);

/** 
 * @brief Allocate memory of single precision complex floating points
 * @param sz  : size_t : size to allocate
 * @return void ptr or NULL
 */
extern void* fpgaf_complex_malloc(size_t sz);

extern fpga_t fpga_test(unsigned N, float2 *inp, float2 *out, bool interleaving);

#endif
