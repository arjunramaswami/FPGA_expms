// Author: Arjun Ramaswami

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>
#define CL_VERSION_2_0
#include <CL/cl_ext_intelfpga.h> // to disable interleaving & transfer data to specific banks - CL_CHANNEL_1_INTELFPGA
#include "CL/opencl.h"

#include "bare.h"
#include "svm.h"
#include "opencl_utils.h"
#include "misc.h"

#ifndef KERNEL_VARS
#define KERNEL_VARS
static cl_platform_id platform = NULL;
static cl_device_id *devices;
static cl_device_id device = NULL;
static cl_context context = NULL;
//static cl_program program = NULL;
static cl_command_queue queue1 = NULL, queue2 = NULL, queue3 = NULL;
static cl_mem d_inData_persist = NULL;

//static int svm_handle;
static int svm_enabled = 0;
#endif

static void queue_setup();
void queue_cleanup();

/** 
 * @brief Allocate memory of double precision complex floating points
 * @param sz  : size_t - size to allocate
 * @param svm : 1 if svm
 * @return void ptr or NULL
 */
void* fpga_complex_malloc(size_t sz){
  if(sz == 0){
    return NULL;
  }
  else{
    return ((double2 *)alignedMalloc(sz));
  }
}

/** 
 * @brief Allocate memory of single precision complex floating points
 * @param sz  : size_t : size to allocate
 * @param svm : 1 if svm
 * @return void ptr or NULL
 */
void* fpgaf_complex_malloc(size_t sz){

  if(sz == 0){
    return NULL;
  }
  return ((float2 *)alignedMalloc(sz));
}

/** 
 * @brief Initialize FPGA
 * @param platform name: string - name of the OpenCL platform
 * @param path         : string - path to binary
 * @param use_svm      : 1 if true 0 otherwise
 * @return 0 if successful 
          -1 Path to binary missing
          -2 Unable to find platform passed as argument
          -3 Unable to find devices for given OpenCL platform
          -4 Failed to create program, file not found in path
          -5 Device does not support required SVM

 */
int fpga_initialize(const char *platform_name, const char *path, bool use_svm){
  cl_int status = 0;

#ifdef VERBOSE
  printf("\tInitializing FPGA ...\n");
#endif

  // Path to binary missing
  if(path == NULL || strlen(path) == 0){
    return -1;
  }

  // Check if this has to be sent as a pointer or value
  // Get the OpenCL platform.
  platform = findPlatform(platform_name);
  // Unable to find given OpenCL platform
  if(platform == NULL){
    return -2;
  }
  // Query the available OpenCL devices.
  cl_uint num_devices;
  devices = getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices);
  // Unable to find device for the OpenCL platform
  if(devices == NULL){
    return -3;
  }

  // use the first device.
  device = devices[0];

  if(use_svm){
    if(!check_valid_svm_device(device)){
      return -5;
    }
    else{
      printf("Supports SVM \n");
      svm_enabled = 1;
    }
  }

  // Create the context.
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
  checkError(status, "Failed to create context");

#ifdef VERBOSE
  printf("\tGetting program binary from path %s ...\n", path);
#endif
  // Create the program.
  /*
  program = getProgramWithBinary(context, &device, 1, path);
  if(program == NULL) {
    fprintf(stderr, "Failed to create program\n");
    fpga_final();
    return -4;
  }

#ifdef VERBOSE
  printf("\tBuilding program ...\n");
#endif
  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");
  */

  return 0;
}

/** 
 * @brief Release FPGA Resources
 */
void fpga_final(){

#ifdef VERBOSE
  printf("\tCleaning up FPGA resources ...\n");
#endif
  /*
  if(program) 
    clReleaseProgram(program);
    */
  if(context)
    clReleaseContext(context);
  free(devices);
}

/**
 * \brief  compute an out-of-place single precision complex 2D-FFT using the BRAM of the FPGA
 * \param  N    : integer pointer to size of FFT2d  
 * \param  inp  : float2 pointer to input data of size [N * N]
 * \param  out  : float2 pointer to output data of size [N * N]
 * \return fpga_t : time taken in milliseconds for data transfers and execution
 */
fpga_t fpga_test(unsigned N, float2 *inp, float2 *out, bool interleaving){
  fpga_t test_time = {0.0, 0.0, 0.0, 0};
  //cl_kernel test_kernel = NULL;

  cl_int status = 0;
  size_t num_pts = N;

  // if N is not a power of 2
  if(inp == NULL || out == NULL || ( (N & (N-1)) !=0)){
    return test_time;
  }

  queue_setup();

  cl_mem_flags flagbuf;
  flagbuf = CL_MEM_READ_WRITE;
   
  // Device memory buffers
  cl_mem d_inData;
  d_inData = clCreateBuffer(context, flagbuf, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");

 // Copy data from host to device
  test_time.pcie_write_t = getTimeinMilliSec();

  status = clEnqueueWriteBuffer(queue1, d_inData, CL_TRUE, 0, sizeof(float2) * num_pts, inp, 0, NULL, NULL);

  status = clFinish(queue1);
  checkError(status, "failed to finish");

  double temp_write = getTimeinMilliSec();
  //printf("Write before - %lf, Write after - %lf \n", test_time.pcie_write_t, temp_write);
  test_time.pcie_write_t = temp_write - test_time.pcie_write_t;
  checkError(status, "Failed to copy data to device");

  /*
  test_kernel = clCreateKernel(program, "test", &status);
  checkError(status, "Failed to create fft2da kernel");

  status = clSetKernelArg(test_kernel, 0, sizeof(cl_mem), (void *)&d_inData);
  checkError(status, "Failed to set fetch kernel arg 0");

  status = clSetKernelArg(test_kernel, 1, sizeof(cl_int), (void *)&how_many);
  checkError(status, "Failed to set fetch kernel arg 1");

  test_time.exec_t = getTimeinMilliSec();
  status = clEnqueueTask(queue1, test_kernel, 0, NULL, NULL);
  checkError(status, "Failed to launch fetch kernel");

  status = clFinish(queue1);
  checkError(status, "failed to finish queue1");
  test_time.exec_t = getTimeinMilliSec() - test_time.exec_t;
  */
  // Copy results from device to host
  test_time.pcie_read_t = getTimeinMilliSec();
  status = clEnqueueReadBuffer(queue1, d_inData, CL_TRUE, 0, sizeof(float2) * num_pts, out, 0, NULL, NULL);

  status = clFinish(queue1);
  checkError(status, "failed to finish reading buffer using PCIe");

  double temp_read = getTimeinMilliSec();
  //printf("Read before - %lf, Read after - %lf \n", test_time.pcie_read_t, temp_read);
  test_time.pcie_read_t = temp_read - test_time.pcie_read_t;
  checkError(status, "Failed to copy data from device");

  queue_cleanup();

  if (d_inData)
  	clReleaseMemObject(d_inData);

  /*
  if(test_kernel) 
    clReleaseKernel(test_kernel);  
    */

  test_time.valid = 1;
  return test_time;
}

int fpga_initialize_withBuf(const char *platform_name, const char *path, bool use_svm, unsigned N){
  cl_int status = 0;

#ifdef VERBOSE
  printf("\tInitializing FPGA ...\n");
#endif

  // Path to binary missing
  if(path == NULL || strlen(path) == 0){
    return -1;
  }

  // Check if this has to be sent as a pointer or value
  // Get the OpenCL platform.
  platform = findPlatform(platform_name);
  // Unable to find given OpenCL platform
  if(platform == NULL){
    return -2;
  }
  // Query the available OpenCL devices.
  cl_uint num_devices;
  devices = getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices);
  // Unable to find device for the OpenCL platform
  if(devices == NULL){
    return -3;
  }

  // use the first device.
  device = devices[0];

  if(use_svm){
    if(!check_valid_svm_device(device)){
      return -5;
    }
    else{
      printf("Supports SVM \n");
      svm_enabled = 1;
    }
  }

  // Create the context.
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
  checkError(status, "Failed to create context");

  cl_mem_flags flagbuf;
  flagbuf = CL_MEM_READ_WRITE;
  size_t num_pts = N;
   
  // Device memory buffers
  d_inData_persist = clCreateBuffer(context, flagbuf, sizeof(float2) * num_pts, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");

  return 0;
}


/**
 * \brief  compute an out-of-place single precision complex 2D-FFT using the BRAM of the FPGA
 * \param  N    : integer pointer to size of FFT2d  
 * \param  inp  : float2 pointer to input data of size [N * N]
 * \param  out  : float2 pointer to output data of size [N * N]
 * \return fpga_t : time taken in milliseconds for data transfers and execution
 */
fpga_t fpga_test_bufPersist(unsigned N, float2 *inp, float2 *out, bool interleaving){
  fpga_t test_time = {0.0, 0.0, 0.0, 0};

  cl_int status = 0;
  size_t num_pts = N;

  // if N is not a power of 2
  if(inp == NULL || out == NULL || ( (N & (N-1)) !=0)){
    return test_time;
  }

  queue_setup();

 // Copy data from host to device
  test_time.pcie_write_t = getTimeinMilliSec();

  status = clEnqueueWriteBuffer(queue1, d_inData_persist, CL_TRUE, 0, sizeof(float2) * num_pts, inp, 0, NULL, NULL);

  status = clFinish(queue1);
  checkError(status, "failed to finish");

  double temp_write = getTimeinMilliSec();
  test_time.pcie_write_t = temp_write - test_time.pcie_write_t;
  checkError(status, "Failed to copy data to device");

  test_time.pcie_read_t = getTimeinMilliSec();
  status = clEnqueueReadBuffer(queue1, d_inData_persist, CL_TRUE, 0, sizeof(float2) * num_pts, out, 0, NULL, NULL);

  status = clFinish(queue1);
  checkError(status, "failed to finish reading buffer using PCIe");

  double temp_read = getTimeinMilliSec();
  test_time.pcie_read_t = temp_read - test_time.pcie_read_t;
  checkError(status, "Failed to copy data from device");

  queue_cleanup();

  test_time.valid = 1;
  return test_time;
}

void fpga_final_withBuf(){
  if (d_inData_persist)
  	clReleaseMemObject(d_inData_persist);

  if(context)
    clReleaseContext(context);
  free(devices);
}


/**
 * \brief Create a command queue for each kernel
 */
void queue_setup(){
  cl_int status = 0;
  // Create one command queue for each kernel.
  queue1 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue1");
  queue2 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue1");
  queue3 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue1");
}

/**
 * \brief Release all command queues
 */
void queue_cleanup() {
  if(queue1) 
    clReleaseCommandQueue(queue1);
  if(queue2) 
    clReleaseCommandQueue(queue2);
  if(queue3) 
    clReleaseCommandQueue(queue3);
}

/**
 * \brief  compute an out-of-place single precision complex 2D-FFT using the BRAM of the FPGA
 * \param  N    : integer pointer to size of FFT2d  
 * \param  inp  : float2 pointer to input data of size [N * N]
 * \param  out  : float2 pointer to output data of size [N * N]
 * \param  how_many : number of batch iterations
 * \return fpga_t : time taken in milliseconds for data transfers and execution
 */
fpga_t nb_pcie_test(unsigned N, float2 *inp, float2 *out, bool interleaving, unsigned how_many){
  fpga_t test_time = {0.0, 0.0, 0.0, 0};
  cl_mem d_inoutData[2];
  cl_int status = 0;

  // if N is not a power of 2
  if(inp == NULL || out == NULL || ( (N & (N-1)) !=0) || (how_many <= 1)){
    return test_time;
  }

  queue_setup();

  // Device Buffers
  d_inoutData[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_1_INTELFPGA, sizeof(float2) * N, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");

  d_inoutData[1] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * N, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");

  cl_event writeEvent[1];

  test_time.exec_t = getTimeinMilliSec();

  clEnqueueWriteBuffer(queue1, d_inoutData[0], CL_TRUE, 0, sizeof(float2) * N, inp, 0, NULL, NULL);
  clFinish(queue1);

  for(size_t i = 1; i < how_many; i++){
    clEnqueueWriteBuffer(queue1, d_inoutData[i%2], CL_FALSE, 0, sizeof(float2) * N, &inp[i * N], 0, NULL, &writeEvent[0]);

    status = clEnqueueReadBuffer(queue2, d_inoutData[(i-1)%2], CL_FALSE, 0, sizeof(float2) * N, &out[(i-1) * N], 0, NULL, &writeEvent[1]);
    checkError(status, "Failed to read");

    clFinish(queue1);
    clFinish(queue2);

    clWaitForEvents(2, writeEvent);
    clReleaseEvent(writeEvent[0]);
    clReleaseEvent(writeEvent[1]);
  }

  status = clEnqueueReadBuffer(queue1, d_inoutData[(how_many-1) % 2], CL_FALSE, 0, sizeof(float2) * N, &out[(how_many - 1) * N], 0, NULL,  &writeEvent[0]);
  checkError(status, "Failed to read");

  clFinish(queue1);
  clWaitForEvents(1, &writeEvent[0]);
  clReleaseEvent(writeEvent[0]);

  test_time.exec_t = getTimeinMilliSec() - test_time.exec_t;
  checkError(status, "Failed to copy data from device");

  queue_cleanup();

  if (d_inoutData[0])
  	clReleaseMemObject(d_inoutData[0]);
  if (d_inoutData[1])
  	clReleaseMemObject(d_inoutData[1]);

  test_time.valid = 1;
  return test_time;
}

/**
 * \brief nonblocking PCIe memory transfer test using event based
 * synchronization
 * \param  N    : size of data
 * \param  inp  : float2 pointer to input data of size [N * N]
 * \param  out  : float2 pointer to output data of size [N * N]
 * \param  how_many : number of batch iterations
 * \return fpga_t : time taken in milliseconds for data transfers and execution
 */
fpga_t nb_event_pcie_test(unsigned N, float2 *inp, float2 *out, bool interleaving, unsigned how_many){
  fpga_t test_time = {0.0, 0.0, 0.0, 0};
  cl_mem d_inoutData[2];
  cl_int status = 0;

  // if N is not a power of 2
  if(inp == NULL || out == NULL || ( (N & (N-1)) !=0) || (how_many <= 1)){
    return test_time;
  }

  queue_setup();

  // Device Buffers
  d_inoutData[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_1_INTELFPGA, sizeof(float2) * N, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");

  d_inoutData[1] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * N, NULL, &status);
  checkError(status, "Failed to allocate input device buffer\n");

  cl_event writeEvent[2], readEvent[2];

  test_time.exec_t = getTimeinMilliSec();

  for(size_t i = 0; i < how_many; i++){
    if(i < 2){
      status = clEnqueueWriteBuffer(queue1, d_inoutData[i%2], CL_FALSE, 0, sizeof(float2) * N, &inp[i * N], 0, NULL, &writeEvent[i]);
      checkError(status, "Failed to write to DDR");
      clFlush(queue1);
    }
    else{
      status = clEnqueueWriteBuffer(queue1, d_inoutData[i%2], CL_FALSE, 0, sizeof(float2) * N, &inp[i * N], 1, &readEvent[i-2], &writeEvent[i]);
      checkError(status, "Failed to write to DDR");
      clFlush(queue1);
    }

    status = clEnqueueReadBuffer(queue2, d_inoutData[i%2], CL_FALSE, 0, sizeof(float2) * N, &out[i * N], 1, &writeEvent[i], &readEvent[i]);
    checkError(status, "Failed to read");
    clFlush(queue2);
  }

  test_time.exec_t = getTimeinMilliSec() - test_time.exec_t;
  checkError(status, "Failed to copy data from device");

  queue_cleanup();

  if (d_inoutData[0])
  	clReleaseMemObject(d_inoutData[0]);
  if (d_inoutData[1])
  	clReleaseMemObject(d_inoutData[1]);

  test_time.valid = 1;
  return test_time;
}
