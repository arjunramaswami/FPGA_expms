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
bool fftf_create_data(float2 *inp, int N){

  if(inp == NULL || N <= 0){
    return false;
  }

  for(int i = 0; i < N; i++){
    inp[i].x = (float)((float)rand() / (float)RAND_MAX);
    inp[i].y = (float)((float)rand() / (float)RAND_MAX);
  }

  return true;
}

/**
 * \brief  create random double precision complex floating point values  
 * \param  inp : pointer to double2 data of size N 
 * \param  N   : number of points in the array
 * \return true if successful
 */
bool fft_create_data(double2 *inp, int N){

  if(inp == NULL || N <= 0 || N > 1024){
    return false;
  }

  for(int i = 0; i < N; i++){
    inp[i].x = (double)((double)rand() / (double)RAND_MAX);
    inp[i].y = (double)((double)rand() / (double)RAND_MAX);
  }

  return true;
}

/**
 * \brief  print configuration chosen to execute on FPGA
 * \param  N: fft size
 * \param  dim: number of dimensions of size
 * \param  iter: number of iterations of each transformation (if BATCH mode)
 */
void print_config(unsigned N, unsigned iter){
  printf("\n------------------------------------------\n");
  printf("Test Configuration: \n");
  printf("--------------------------------------------\n");
  printf("Type               = Complex to Complex\n");
  printf("Points             = %d\n", N);
  printf("Iterations         = %d\n", iter);
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

  double gpoints_per_sec = (N  / (exec * 1e-3)) * 1e-9;
  double gBytes_per_sec = 0.0;

  gBytes_per_sec =  gpoints_per_sec * 8; // bytes

  printf("\n------------------------------------------\n");
  printf("Measurements \n");
  printf("--------------------------------------------\n");
  printf("Points             = %d\n", N);
  printf("Iterations         = %d\n", iter);

  printf("%s", iter>1 ? "Average Measurements of iterations\n":"");
  printf("PCIe Write         = %.2lfms\n", pcie_write);
  printf("Kernel Execution   = %.2lfms\n", exec);
  printf("PCIe Read          = %.2lfms\n", pcie_read);
  printf("Total              = %.2lfms\n", pcie_read + exec + pcie_write);
  printf("API runtime        = %.2lfms\n", avg_api_time);
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