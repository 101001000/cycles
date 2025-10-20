#include <sys/sysinfo.h>
#include "device/simple/device_impl.h"
#include "kernel/device/simple/globals.h"
#include "device/simple/queue.h"
#include "bvh/bvh2.h"
#include <portableRT/portableRT.hpp>
#include "scene/geometry.h"
#include "scene/mesh.h"
#include <dlfcn.h>

CCL_NAMESPACE_BEGIN

extern "C" int kernel_force_init();
static int _force = kernel_force_init();


SimpleDevice::SimpleDevice(const DeviceInfo &info, Stats &stats, Profiler &profiler, bool headless) : GPUDevice(info, stats, profiler, headless) {

    prt::kernelapi_init({{"kernel_globals", sizeof(KernelGlobalsGPU)}});

    std::cout << "test2" << std::endl;

    std::cout << "available backends: " << std::endl;
    for (int i = 0; i < prt::available_backends().size(); i++) {
        std::cout << i << ": " << prt::available_backends()[i]->name() << " " << std::endl;
    }
    std::cout << "select backend: ";
    int idx;
    std::cin >> idx;
    prt::select_backend(prt::available_backends()[idx]);
    m_backend = prt::selected_backend;
    std::cout << "selected backend: " << prt::selected_backend->name() << std::endl;
    m_backend->global_alloc("kernel_globals", sizeof(KernelGlobalsGPU));

    
    

    std::cout << "showing all kernels:" << std::endl;
    for (auto [name, fn] : prt::kernels_) {
        std::cout << "Listing kernel " << name << std::endl;
    }

    for (auto [data, size] : prt::embeded_kernels_) {
        std::cout << "Listing embeded kernel " << size << std::endl;
    }

    std::cout << "test" << std::endl;

    for (auto [name, size] : prt::global_vars_) {
        std::cout << "Listing global " << name << " of size " << size << std::endl;
    }

}

SimpleDevice::~SimpleDevice() {
    m_backend->global_free("kernel_globals");
}


void SimpleDevice::global_free(device_memory &mem)
{
  if (mem.device_pointer) {
    m_backend->device_free((void *)mem.device_pointer); 
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

    std::cout << "const copying " << name << " to device (" << size << " bytes)" << std::endl;


    KernelGlobalsGPU kg_host;
    m_backend->global_copy_from("kernel_globals", &kg_host, sizeof(KernelGlobalsGPU));

    void* ptr = host;

    if (strcmp(name, "data") == 0) {
        kg_host.__data = (KernelData *)ptr;
    } else if (strcmp(name, "integrator_state") == 0) {
        kg_host.integrator_state = (IntegratorStateGPU *)ptr;
    }

    #define KERNEL_DATA_ARRAY(t, nm)                                                     \
    if (strcmp(name, #nm) == 0) {                                                      \
      kg_host.__##nm = (const t *)ptr;                                          \
    }
    KERNEL_DATA_ARRAY(int, object_id)
    #include "kernel/data_arrays.h"
    #undef KERNEL_DATA_ARRAY

    m_backend->global_copy_to("kernel_globals", &kg_host, sizeof(KernelGlobalsGPU));
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
    } else if (mem.type == MEM_TEXTURE) {
        tex_free((device_texture &)mem);
    } else if (mem.device_pointer) {
        if (mem.type == MEM_DEVICE_ONLY) {
            free((void*)mem.device_pointer);   
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

std::string geometry_type_name(Geometry::Type type){
    if (type == Geometry::Type::MESH) {
        return "mesh";
    }
    if (type == Geometry::Type::HAIR) {
        return "hair";
    }
    if (type == Geometry::Type::VOLUME) {
        return "volume";
    }
    if (type == Geometry::Type::POINTCLOUD) {
        return "pointcloud";
    }
    if (type == Geometry::Type::LIGHT) {
        return "light";
    }
    return "unknown";
}

void SimpleDevice::build_bvh(BVH *bvh, Progress &progress, bool refit){
    std::cout << "building bvh!" << std::endl;

    std::vector<std::array<float, 9>> tris;
    std::vector<int> object_ids;

    for (Geometry *geometry : bvh->geometry) {
        std::cout << "analizando geometría" << geometry_type_name(geometry->geometry_type) << " " << std::endl;

        if (geometry->is_mesh()) {
            Mesh *mesh = static_cast<Mesh *>(geometry);
            for (size_t i = 0; i < mesh->num_triangles(); ++i) {
                Mesh::Triangle tri = mesh->get_triangle(i);
                float3 v0 = mesh->verts[tri.v[0]];
                float3 v1 = mesh->verts[tri.v[1]];
                float3 v2 = mesh->verts[tri.v[2]];
                tris.push_back({v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z});            
                object_ids.push_back(geometry->index);
            }
        }
    }

    void* object_id_ptr = malloc(object_ids.size() * sizeof(int));
    memcpy(object_id_ptr, object_ids.data(), object_ids.size() * sizeof(int));
    const_copy_to("object_id", object_id_ptr, object_ids.size() * sizeof(int));

    std::cout << "building..." << std::endl;
    prt::selected_backend->set_tris(tris);
    std::cout << "built!" << std::endl;

}

CCL_NAMESPACE_END