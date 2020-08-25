# Arjun Ramaswami
cmake_minimum_required(VERSION 3.10)
project(expms VERSION 0.1
            DESCRIPTION "Experiments code that make use of the api"
            LANGUAGES C CXX)

set(examples new_data reuse_data)

# create a target for each of the example 
foreach(example ${examples})

  add_executable(${example} ${example}.c
                  common/helper.c)
                  #common/verify_fftw.c)

  target_compile_options(${example}
      PRIVATE -Wall -Werror)
      
  target_include_directories(${example}
      PRIVATE  ${IntelFPGAOpenCL_INCLUDE_DIRS}
                "${CMAKE_SOURCE_DIR}/extern/argparse"
                common)
    
  target_link_libraries(${example}
      PRIVATE ${IntelFPGAOpenCL_LIBRARIES} bare argparse m)

endforeach()