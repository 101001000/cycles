#ifdef OPTIX_KERNEL
  #include "kernel/device/optix/compat.h"
  #define ccl_gpu_block_dim_x 1
  #define ccl_gpu_thread_idx_x 0
  #define ccl_gpu_warp_size 1
  #define ccl_gpu_block_idx_x 0
  #define ccl_gpu_ballot 1
  #define ccl_gpu_thread_mask 1
  #define ccl_gpu_syncthreads void
  #define ccl_gpu_block_idx_x global_idx
  #define ccl_gpu_ballot(predicate) (predicate ? 1 : 0)
  #define ccl_gpu_kernel_call(x) x
  #define ccl_gpu_thread_mask(thread_warp) \
    ((thread_warp) >= 1 ? 1 : 0)  // Máscara para 1 hilo
  #define ccl_gpu_global_id_x() (ccl_gpu_block_idx_x * ccl_gpu_block_dim_x + \
    ccl_gpu_thread_idx_x)
  #define ccl_gpu_shared

  //__constant__ int warp_offset[128 * 10 + 1];
  #undef __KERNEL_OPTIX__
  #define __KERNEL_GPU__
  #define __KERNEL_SIMPLE__
#endif

#ifdef CPU_KERNEL
  #include "kernel/device/simple/compat.h"
  int warp_offset[1024 * 10 + 1];
#endif

#include "kernel/device/simple/config.h"
#include "kernel/device/simple/globals.h"

#define PRT_GLOBALS PRT_GVAR(kernel_globals, KernelParamsSimple) PRT_GVAR(warp_offset, int*)

#include <portableRT/portableRT.hpp>

#include "kernel/device/gpu/image.h"
#include "kernel/device/gpu/kernel.h"


static auto run = [](){
  //std::cout << "KERNEL.CPP!!!" << std::endl;
  return 0;
}();


extern "C" int kernel_force_init();
extern "C" int kernel_force_init() { return run; }

