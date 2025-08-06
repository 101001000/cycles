#include <sys/sysinfo.h>
#include "device/simple/device_impl.h"
#include "kernel/device/simple/globals.h"
#include "device/simple/queue.h"
#include "bvh/bvh2.h"

CCL_NAMESPACE_BEGIN

SimpleDevice::SimpleDevice(const DeviceInfo &info, Stats &stats, Profiler &profiler, bool headless) : GPUDevice(info, stats, profiler, headless) {
    kernel_globals.idx = (int*)malloc(sizeof(int));
    kernel_globals.dim = (int*)malloc(sizeof(int));
    kernel_globals.bid = (int*)malloc(sizeof(int));
    simple_set_idx(kernel_globals.idx, kernel_globals.dim, kernel_globals.bid);
}

SimpleDevice::~SimpleDevice() {
    free(kernel_globals.idx);
    free(kernel_globals.dim);
    free(kernel_globals.bid);

    //free(const_cast<void*>(reinterpret_cast<const void*>(kernel_globals.__data)));
    //free(kernel_globals.integrator_state);

    //#define KERNEL_DATA_ARRAY(t, nm) \
    //free(const_cast<void*>(reinterpret_cast<const void*>(kernel_globals.__##nm)));
//#include "kernel/data_arrays.h"
//#undef KERNEL_DATA_ARRAY
}


void SimpleDevice::global_free(device_memory &mem)
{
  if (mem.device_pointer) {
    free((void *)mem.device_pointer); 
    mem.device_pointer = 0;
    stats.mem_free(mem.device_size);
    mem.device_size = 0;
  }
}

void SimpleDevice::tex_alloc(device_texture &mem)
{
  VLOG_WORK << "Texture allocate: " << mem.name << ", "
            << string_human_readable_number(mem.memory_size()) << " bytes. ("
            << string_human_readable_size(mem.memory_size()) << ")";

  mem.device_pointer = (device_ptr)mem.host_pointer;
  mem.device_size = mem.memory_size();
  stats.mem_alloc(mem.device_size);

  const uint slot = mem.slot;
  if (slot >= texture_info.size()) {
    /* Allocate some slots in advance, to reduce amount of re-allocations. */
    texture_info.resize(slot + 128);
  }

  texture_info[slot] = mem.info;
  texture_info[slot].data = (uint64_t)mem.host_pointer;
  need_texture_info = true;
}


void SimpleDevice::tex_free(device_texture &mem)
{
  if (mem.device_pointer) {
    mem.device_pointer = 0;
    stats.mem_free(mem.device_size);
    mem.device_size = 0;
    need_texture_info = true;
  }
}

BVHLayoutMask SimpleDevice::get_bvh_layout_mask(const uint kernel_features) const {return BVH_LAYOUT_SIMPLE;}
void SimpleDevice::const_copy_to(const char *name, void *host, const size_t size){

    std::cout << "copying " << name << " to device (" << size << " bytes)" << std::endl;

    //void* ptr = malloc(size);
    //memcpy(ptr, host, size);
    void* ptr = host;

    if (strcmp(name, "data") == 0) { kernel_globals.__data = (KernelData *)ptr;
        simple_set_data(ptr);
        return; }
    if (strcmp(name, "integrator_state") == 0) {
        kernel_globals.integrator_state = (IntegratorStateGPU *)ptr;
        simple_set_integrator_state(ptr);   
        return;
    }
    #define KERNEL_DATA_ARRAY(t, nm)                                                     \
    if (strcmp(name, #nm) == 0) {                                                      \
      kernel_globals.__##nm = (const t *)ptr;                                          \
      simple_set_data_array(#nm, ptr);                                                 \
      return;                                                                          \
    }
    #include "kernel/data_arrays.h"
    #undef KERNEL_DATA_ARRAY

    assert(false && "nombre desconocido");
}

void SimpleDevice::global_alloc(device_memory &mem)
{
    void *ptr = malloc(mem.memory_size());
    memcpy(ptr, mem.host_pointer, mem.memory_size());

    mem.device_pointer = (device_ptr)ptr;
    mem.device_size    = mem.memory_size();
    stats.mem_alloc(mem.device_size);

    const_copy_to(mem.name, ptr, mem.memory_size());   // size no importa aquí
}
void SimpleDevice::mem_alloc(device_memory &mem){
    if (mem.type == MEM_DEVICE_ONLY) {
        void *data = malloc(mem.memory_size());
        mem.device_pointer = (device_ptr)data;
    }
    else {
        assert(!(mem.host_pointer == nullptr && mem.memory_size() > 0));
        mem.device_pointer = (device_ptr)mem.host_pointer;
    }
    mem.device_size = mem.memory_size();
}
void SimpleDevice::mem_copy_to(device_memory &mem){
    if (mem.type == MEM_GLOBAL) {
        global_free(mem);
        global_alloc(mem);
    }
    else if (mem.type == MEM_TEXTURE) {
        tex_free((device_texture &)mem);
        tex_alloc((device_texture &)mem);
    }
    else {
        if (!mem.device_pointer) {
            mem_alloc(mem);
        }
        /* copy is no-op */
    }
}
void SimpleDevice::mem_move_to_host(device_memory &mem){

}
void SimpleDevice::mem_zero(device_memory &mem){
    if (!mem.device_pointer) {
        mem_alloc(mem);
    }
    if (mem.device_pointer) {
        memset((void *)mem.device_pointer, 0, mem.memory_size());
    }
}
void SimpleDevice::mem_free(device_memory &mem){
    if (mem.type == MEM_GLOBAL) {
        global_free(mem);
    }
    else if (mem.type == MEM_TEXTURE) {
        tex_free((device_texture &)mem);
    }
    else if (mem.device_pointer) {
        if (mem.type == MEM_DEVICE_ONLY) {
          util_aligned_free((void *)mem.device_pointer, mem.memory_size());
        }
        mem.device_pointer = 0;
        mem.device_size = 0;
    }
}
void SimpleDevice::mem_copy_from(device_memory &mem, const size_t y, size_t w, const size_t h, size_t elem){

}
void SimpleDevice::get_device_memory_info(size_t &total, size_t &free){
    struct sysinfo info;
    if (sysinfo(&info) != 0) { 
        total = free = 0;
        return;
    }
    total = static_cast<size_t>(info.totalram) * info.mem_unit;
    free  = static_cast<size_t>(info.freeram) * info.mem_unit;
}
bool SimpleDevice::alloc_device(void *&device_pointer, const size_t size){
    device_pointer = malloc(size);
    return device_pointer != nullptr;
}
void SimpleDevice::free_device(void *device_pointer){
    if(device_pointer){
        free(device_pointer);
    }
}
bool SimpleDevice::shared_alloc(void *&shared_pointer, const size_t size){
    return alloc_device(shared_pointer, size);
}
void SimpleDevice::shared_free(void *shared_pointer){
    free_device(shared_pointer);
}
void *SimpleDevice::shared_to_device_pointer(const void *shared_pointer){
    return (void *)shared_pointer;
}
void SimpleDevice::copy_host_to_device(void *device_pointer, void *host_pointer, const size_t size){
    memcpy(device_pointer, host_pointer, size);
}

unique_ptr<DeviceQueue> SimpleDevice::gpu_queue_create() {
    return make_unique<SimpleDeviceQueue>(this);
}

void SimpleDevice::build_bvh(BVH *bvh, Progress &progress, bool refit){
    std::cout << "building bvh!" << std::endl;

}

CCL_NAMESPACE_END