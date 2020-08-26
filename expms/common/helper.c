// Author: Arjun Ramaswami

#define _POSIX_C_SOURCE 199309L  
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "helper.h"
#include <time.h>
#include <sys/time.h>
#include <math.h>
#define _USE_MATH_DEFINES

/**
 * \brief  create random single precision complex floating point values  
 * \param  inp : pointer to float2 data of size N 
 * \param  N   : number of points in the array
 * \return true if successful
 */
bool create_data(float2 *inp, unsigned N){

  if(inp == NULL || N <= 0){
    return false;
  }

  for(size_t i = 0; i < N; i++){
    inp[i].x = (float)((float)rand() / (float)RAND_MAX);
    inp[i].y = (float)((float)rand() / (float)RAND_MAX);
  }

  return true;
}

/**
 * \brief  print configuration chosen to execute on FPGA
 * \param  N: fft size
 * \param  dim: number of dimensions of size
 * \param  iter: number of iterations of each transformation (if BATCH mode)
 */
void print_config(unsigned N, unsigned iter, bool interleaving){
  printf("\n------------------------------------------\n");
  printf("Test Configuration: \n");
  printf("--------------------------------------------\n");
  printf("Type               = Complex to Complex\n");
  printf("Points             = %d\n", N);
  printf("Iterations         = %d\n", iter);
  printf("Interleaving       = %d\n", iter);
  printf("--------------------------------------------\n\n");
}

/**
 * \brief  print time taken for fpga and fftw runs to a file
 * \param  total_api_time: time taken to call iter times the host code
 * \param  timing: kernel execution and pcie transfer timing 
 * \param  N: fft size
 * \param  iter: number of iterations of each transformation (if BATCH mode)
 */
void display_measures(double total_api_time, double pcie_rd, double pcie_wr, double exec_t, unsigned N, unsigned iter){

  double avg_api_time = 0.0;

  if (total_api_time != 0.0){
    avg_api_time = total_api_time / iter;
  }

  double pcie_read = pcie_rd / iter;
  double pcie_write = pcie_wr / iter;
  double exec = exec_t / iter;
  unsigned data_sz = N * 8;
  double pcie_rd_bandwidth = data_sz * 1e-9 / (pcie_read * 1e-3);
  double pcie_wr_bandwidth = data_sz  * 1e-9 / (pcie_write * 1e-3);

  printf("\n------------------------------------------\n");
  printf("Measurements \n");
  printf("--------------------------------------------\n");
  printf("Iterations             = %d\n", iter);
  printf("Points                 = %d\n", N);
  printf("Data Size              = %u Bytes\n", data_sz);
  printf("PCIe Write Latency     = %.5lfms\n", pcie_write);
  printf("PCIe Read Latency      = %.5lfms\n", pcie_read);
  printf("PCIe Write Bandwidth   = %.5lf GB/s\n", pcie_wr_bandwidth);
  printf("PCIe Read Bandwidth    = %.5lf GB/s\n", pcie_rd_bandwidth);
  printf("Average Exec Time      = %.5lfms\n", exec);
  printf("Average API Time       = %.5lfms\n", avg_api_time);
}

/**
 * \brief  compute walltime in milliseconds
 * \return time in milliseconds
 */
double getTimeinMilliseconds(){
   struct timespec a;
   if(clock_gettime(CLOCK_MONOTONIC, &a) != 0){
     fprintf(stderr, "Error in getting wall clock time \n");
     exit(EXIT_FAILURE);
   }
   return (double)(a.tv_nsec) * 1.0e-6 + (double)(a.tv_sec) * 1.0E3;
}