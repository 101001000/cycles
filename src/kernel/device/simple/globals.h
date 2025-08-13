#pragma once

#include <iostream>

#include "kernel/types.h"

#include "kernel/integrator/state.h"
#include "kernel/util/profiler.h"

#include "util/color.h"
#include "util/texture.h"

CCL_NAMESPACE_BEGIN

struct IntegratorStateGPU;

struct KernelGlobalsGPU {

    #define KERNEL_DATA_ARRAY(type, name) const type *__##name = nullptr;
    KERNEL_DATA_ARRAY(int, object_id)
    #include "kernel/data_arrays.h"
    #undef KERNEL_DATA_ARRAY
      const KernelData *__data;
      IntegratorStateGPU *integrator_state;
      int* idx;
      int* dim;
      int* bid;
};

using KernelGlobals = ccl_global KernelGlobalsGPU *ccl_restrict;

extern "C" void simple_set_integrator_state(void *ptr);
extern "C" void simple_set_data_array(const char *name, const void *ptr);
extern "C" void simple_set_data(void *ptr);
extern "C" void simple_set_idx(int *idx, int *dim, int *bid);


extern "C" __attribute__((visibility("default"))) KernelGlobalsGPU kernel_globals;

template<typename T>
ccl_device_inline const T &kernel_data_fetch_dbg_ref(const char *nm,
                                                    const T     *base,
                                                    size_t       i)
{
  if(static_cast<int>(i) < 0){
    std::cout << "WRONG ACCESS [device] " << nm << " = " << static_cast<const void *>(base) << " idx " << i
            << std::endl;
    return base[0];
  }
  //std::cout << "[device] " << nm << " = " << static_cast<const void *>(base) << " idx " << i
  //          << std::endl;
  return base[i];            
}

#define kernel_data (*(kernel_globals.__data))
#define kernel_data_fetch(name, index) \
  kernel_data_fetch_dbg_ref(#name, kernel_globals.__##name, (index))
#define kernel_data_array(name) (kernel_globals.__##name)
#define kernel_integrator_state (*(kernel_globals.integrator_state))

CCL_NAMESPACE_END