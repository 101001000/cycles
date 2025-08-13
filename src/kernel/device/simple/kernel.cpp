#include "kernel/device/simple/config.h"
#include "kernel/device/simple/globals.h"

#include "kernel/device/gpu/image.h"
#include "kernel/device/gpu/kernel.h"

ccl_gpu_kernel(GPU_KERNEL_BLOCK_NUM_THREADS, GPU_KERNEL_MAX_REGISTERS)
    ccl_gpu_kernel_signature(integrator_intersect_closest2,
                             const ccl_global int *path_index_array,
                             ccl_global float *render_buffer,
                             const int work_size)
{
  std::vector<int> states;

  for(int i = 0; i < work_size; i++) {
    const int global_index = i;

    if (ccl_gpu_kernel_within_bounds(global_index, work_size)) {
      const int state = (path_index_array) ? path_index_array[global_index] : global_index;
      states.push_back(state);
    }
  }

  ccl_gpu_kernel_call(integrator_intersect_closest2(nullptr, states, render_buffer));
}
ccl_gpu_kernel_postfix

extern "C" void simple_set_integrator_state(void *ptr)
{
    kernel_globals.integrator_state = static_cast<IntegratorStateGPU*>(ptr);
}

extern "C" void simple_set_data(void *ptr)
{
    kernel_globals.__data = static_cast<KernelData*>(ptr);
}
extern "C" void simple_set_data_array(const char *name, const void *ptr)
{
#define KERNEL_DATA_ARRAY(t, nm)            \
  if (strcmp(name, #nm) == 0) {             \
    kernel_globals.__##nm = (const t *)ptr; \
    return;                                 \
  }
KERNEL_DATA_ARRAY(int, object_id)
#include "kernel/data_arrays.h"
#undef KERNEL_DATA_ARRAY
}
extern "C" __attribute__((visibility("default")))
KernelGlobalsGPU kernel_globals{};
extern "C" void simple_set_idx(int *idx, int *dim, int *bid){
    kernel_globals.idx = idx;
    kernel_globals.dim = dim;
    kernel_globals.bid = bid;
}
int warp_offset[GPU_PARALLEL_ACTIVE_INDEX_DEFAULT_BLOCK_SIZE * 10 + 1];