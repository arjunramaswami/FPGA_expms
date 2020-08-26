//  Author: Arjun Ramaswami

#include <stdio.h>
#include <stdlib.h> // EXIT_FAILURE
#include <math.h>
#include <stdbool.h>

#include "CL/opencl.h"
#include "bare.h"

#include "argparse.h"
#include "helper.h"

static const char *const usage[] = {
    "bin/host [options]",
    NULL,
};

int main(int argc, const char **argv) {
  unsigned N = 1, iter = 1; 
  int use_svm = 0;
  bool interleaving = false;
  char *path = "test.aocx";
  const char *platform;
  
  double avg_rd = 0.0, avg_wr = 0.0, avg_exec = 0.0;
  double total_api_time = 0.0;
  bool status = true, use_emulator = false;

  struct argparse_option options[] = {
    OPT_HELP(),
    OPT_GROUP("Basic Options"),
    OPT_INTEGER('n',"n", &N, "Data Size"),
    OPT_INTEGER('i',"iter", &iter, "Iterations"),
    OPT_BOOLEAN('t',"interleaving", &interleaving, "Use burst interleaving in case of BRAM designs"),
    OPT_STRING('p', "path", &path, "Path to bitstream"),
    OPT_BOOLEAN('e', "emu", &use_emulator, "Use emulator"),
    OPT_END(),
  };

  struct argparse argparse;
  argparse_init(&argparse, options, usage, 0);
  argparse_describe(&argparse, "Experimenting on FPGA", "Data size and path are mandatory, default number of iterations is 1");
  argc = argparse_parse(&argparse, argc, argv);

  // Print to console the configuration chosen to execute during runtime
  print_config(N, iter, interleaving);

  if(use_emulator){
    platform = "Intel(R) FPGA Emulation Platform for OpenCL(TM)";
    //platform = "Intel(R) FPGA";
  }
  else{
    platform = "Intel(R) FPGA SDK for OpenCL(TM)";
    //platform = "Intel(R) FPGA";
  }
  
  int isInit = fpga_initialize(platform, path, use_svm);
  if(isInit != 0){
    fprintf(stderr, "FPGA initialization error %d\n", isInit);
    return EXIT_FAILURE;
  }

  // create and use same data every iteration
  size_t inp_sz = sizeof(float2) * N;
  float2 *inp = (float2*)fpgaf_complex_malloc(inp_sz);
  float2 *out = (float2*)fpgaf_complex_malloc(inp_sz);

  status = create_data(inp, N);

  for(size_t i = 0; i < iter; i++){
    fpga_t timing = {0.0, 0.0, 0.0, 0};
    double temp_timer = 0.0;

    if(!status){
      fprintf(stderr, "Error in Data Creation \n");
      free(inp);
      free(out);
      return EXIT_FAILURE;
    }

    temp_timer = getTimeinMilliseconds();
    timing = fpga_test(N, inp, out, interleaving);
    total_api_time += getTimeinMilliseconds() - temp_timer;

    /*
    if(!verify_output(out, inp, N)){
      fprintf(stderr, "Verification Failed \n");
      free(inp);
      free(out);
      return EXIT_FAILURE;
    }
    */

    if(timing.valid == 0){
      fprintf(stderr, "Invalid execution, timing found to be 0\n");
      free(inp);
      free(out);
      return EXIT_FAILURE;
    }

    avg_rd += timing.pcie_read_t;
    avg_wr += timing.pcie_write_t;
    avg_exec += timing.exec_t;

    printf("Iter: %lu\n", i);
    printf("\tPCIe Rd: %lfms\n", timing.pcie_read_t);
    printf("\tKernel: %lfms\n", timing.exec_t);
    printf("\tPCIe Wr: %lfms\n\n", timing.pcie_write_t);
            
  }  // iter

  // destroy FFT input and output
  free(inp);
  free(out);

  // destroy fpga state
  fpga_final();

  // display performance measures
  display_measures(total_api_time, avg_rd, avg_wr, avg_exec, N, iter);

  return EXIT_SUCCESS;
}
