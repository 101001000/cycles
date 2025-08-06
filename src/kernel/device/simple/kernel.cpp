#include "kernel/device/simple/config.h"
#include "kernel/device/simple/globals.h"

#include "kernel/device/gpu/image.h"
#include "kernel/device/gpu/kernel.h"

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