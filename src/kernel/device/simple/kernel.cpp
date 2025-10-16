
#include "kernel/device/simple/config.h"
#include "kernel/device/simple/globals.h"

#define PRT_GLOBALS PRT_GVAR(kernel_globals, KernelGlobalsGPU)

#include <portableRT/portableRT.hpp>

#include "kernel/device/gpu/image.h"
#include "kernel/device/gpu/kernel.h"


static auto run = [](){
  std::cout << "KERNEL.CPP!!!" << std::endl;
  return 0;
}();


extern "C" int kernel_force_init();
extern "C" int kernel_force_init() { return run; }


int warp_offset[GPU_PARALLEL_ACTIVE_INDEX_DEFAULT_BLOCK_SIZE * 10 + 1];