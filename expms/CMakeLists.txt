# Arjun Ramaswami
cmake_minimum_required(VERSION 3.10)
project(expms VERSION 0.1
            DESCRIPTION "Experiments code that make use of the api"
            LANGUAGES C CXX)

set(examples newdata_newmem newdata_newmem_samedevbuf newdata_samemem newdata_samemem_samedevbuf reusedata_samemem nonblocking_pcietest)

# create a target for each of the example 
foreach(example ${examples})

  add_executable(${example} ${example}.c
                  common/helper.c)
                  #common/verify_fftw.c)

  target_compile_options(${example}
      PRIVATE -Wall -Werror)
      
  target_include_directories(${example}
      PRIVATE  ${IntelFPGAOpenCL_INCLUDE_DIRS}
                "${argparse_SOURCE_DIR}"
                common)
    
  target_link_libraries(${example}
      PRIVATE ${IntelFPGAOpenCL_LIBRARIES} bare argparse m)

endforeach()