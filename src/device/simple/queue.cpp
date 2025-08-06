#include "device/simple/queue.h"
#include "device/simple/device_impl.h"

struct KernelWorkTile;

extern void simple_integrator_init_from_camera(KernelWorkTile *tiles,
    int                    num_tiles,
    float                 *render_buffer,
    int                    max_tile_work_size);

extern void simple_integrator_reset(const int num_states);

extern void simple_integrator_intersect_closest(const int *path_index_array,
    float *render_buffer,
    const int work_size);

extern void simple_prefix_sum(int *counter, int *prefix_sum, const int num_values);

extern void simple_integrator_sorted_paths_array(const int num_states, const int num_states_limit, int *indices, int *num_indices, int *key_counter, int *key_prefix_sum, const int kernel_index);

extern void simple_integrator_shade_surface(const int *path_index_array,
                                            float *render_buffer,
                                            const int work_size);


extern void simple_integrator_shade_background(const int *path_index_array,
                                            float *render_buffer,
                                            const int work_size);

extern void simple_integrator_queued_paths_array(const int num_states, int *indices, int *num_indices, const int kernel_index);

extern void simple_integrator_compact_paths_array(const int num_states, int *indices, int *num_indices, const int num_active_paths);

extern void simple_integrator_terminated_paths_array(const int num_states, int *indices, int *num_indices, const int indices_offset);

extern void simple_integrator_compact_states(const int *active_terminated_states, const int active_states_offset, const int terminated_states_offset, const int work_size);

extern void simple_adaptive_sampling_convergence_check(float *render_buffer, const int sx, const int sy, const int sw, const int sh, const float threshold, const int reset, const int offset, const int stride, uint *num_active_pixels);

CCL_NAMESPACE_BEGIN


SimpleDeviceQueue::SimpleDeviceQueue(SimpleDevice *device) : DeviceQueue(device), device(device) {}
SimpleDeviceQueue::~SimpleDeviceQueue() {}

int SimpleDeviceQueue::num_concurrent_states(const size_t state_size) const { return 1000; }
int SimpleDeviceQueue::num_concurrent_busy_states(const size_t state_size) const { return 1; }
void SimpleDeviceQueue::init_execution() {
    debug_init_execution();
}

template<typename T>
inline T get_scalar(const void *p)
{
    return *reinterpret_cast<const T *>(p);
}
template<typename T>
inline T *get_pointer(const void *p)
{
  return *reinterpret_cast<T * const *>(p);
}

std::string type_to_string(DeviceKernelArguments::Type type) {
    switch(type) {
        case DeviceKernelArguments::Type::POINTER: return "pointer";
        case DeviceKernelArguments::Type::INT32: return "int32";
        case DeviceKernelArguments::Type::FLOAT32: return "float32";
        case DeviceKernelArguments::Type::KERNEL_FILM_CONVERT: return "kernel_film_convert";
        case DeviceKernelArguments::Type::HIPRT_GLOBAL_STACK: return "hiprt_global_stack";
        default: return "unknown";
    }
}

bool SimpleDeviceQueue::enqueue(DeviceKernel kernel, const int work_size, const DeviceKernelArguments &args) {

    debug_enqueue_begin(kernel, work_size);

    const int dim = 1; // Threads per block
    const int num_blocks = (work_size + dim - 1) / dim;
    
    // Create local copies of global variables
    int local_idx = *kernel_globals.idx;
    int local_dim = *kernel_globals.dim;
    int local_bid = *kernel_globals.bid;
    
    for (int b = 0; b < num_blocks; ++b) {
        local_bid = b;
        
        for (int t = 0; t < dim; ++t) {
            const int g = b * dim + t;
            if (g >= work_size) break;
            
            local_idx = t;
            
            // Update global state
            *kernel_globals.idx = local_idx;
            *kernel_globals.dim = dim;
            *kernel_globals.bid = local_bid;
    
            switch(kernel) {
                case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA: {
                    auto *tiles         = get_pointer<ccl::KernelWorkTile>(args.values[0]);
                    int   num_tiles     = get_scalar<int>(args.values[1]);
                    auto *render_buffer = get_pointer<float>(args.values[2]);
                    int   max_tile_work = get_scalar<int>(args.values[3]);
                    ::simple_integrator_init_from_camera(
                        reinterpret_cast<::KernelWorkTile *>(tiles),
                        num_tiles,
                        render_buffer,
                        max_tile_work);
                    break;
                }
                case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_RESET: {
                    int num_states = get_scalar<int>(args.values[0]);
                    ::simple_integrator_reset(num_states);
                    break;
                }
                case DeviceKernel::DEVICE_KERNEL_PREFIX_SUM: {
                    int *counter = get_pointer<int>(args.values[0]);
                    int *prefix_sum = get_pointer<int>(args.values[1]);
                    int num_values = get_scalar<int>(args.values[2]);
                    ::simple_prefix_sum(counter, prefix_sum, num_values);
                    break;
                }
                case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST: {
                    int *path_index_array = get_pointer<int>(args.values[0]);
                    float *render_buffer = get_pointer<float>(args.values[1]);
                    ::simple_integrator_intersect_closest(path_index_array, render_buffer, work_size);
                    break;
                }
                case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_SORTED_PATHS_ARRAY: {
                    int num_states = get_scalar<int>(args.values[0]);
                    int num_states_limit = get_scalar<int>(args.values[1]);
                    int *indices = get_pointer<int>(args.values[2]);
                    int *num_indices = get_pointer<int>(args.values[3]);
                    int *key_counter = get_pointer<int>(args.values[4]);
                    int *key_prefix_sum = get_pointer<int>(args.values[5]);
                    int kernel_index = get_scalar<int>(args.values[6]);
                    ::simple_integrator_sorted_paths_array(num_states, num_states_limit, indices, num_indices, key_counter, key_prefix_sum, kernel_index);
                    break;
                }
                case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE: {
                    int *path_index_array = get_pointer<int>(args.values[0]);
                    float *render_buffer = get_pointer<float>(args.values[1]);
                    ::simple_integrator_shade_surface(path_index_array, render_buffer, work_size);
                    break;
                }
                case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND: {
                    int *path_index_array = get_pointer<int>(args.values[0]);
                    float *render_buffer = get_pointer<float>(args.values[1]);
                    ::simple_integrator_shade_background(path_index_array, render_buffer, work_size);
                    break;
                }
                case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_QUEUED_PATHS_ARRAY: {
                    int num_states = get_scalar<int>(args.values[0]);
                    int *indices = get_pointer<int>(args.values[1]);
                    int *num_indices = get_pointer<int>(args.values[2]);
                    int kernel_index = get_scalar<int>(args.values[3]);
                    ::simple_integrator_queued_paths_array(num_states, indices, num_indices, kernel_index);
                    break;
                }
                case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_COMPACT_PATHS_ARRAY: {
                    int num_states = get_scalar<int>(args.values[0]);
                    int *indices = get_pointer<int>(args.values[1]);
                    int *num_indices = get_pointer<int>(args.values[2]);
                    int num_active_paths = get_scalar<int>(args.values[3]);
                    ::simple_integrator_compact_paths_array(num_states, indices, num_indices, num_active_paths);
                    break;
                }
                case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_TERMINATED_PATHS_ARRAY: {
                    int num_states = get_scalar<int>(args.values[0]);
                    int *indices = get_pointer<int>(args.values[1]);
                    int *num_indices = get_pointer<int>(args.values[2]);
                    int indices_offset = get_scalar<int>(args.values[3]);
                    ::simple_integrator_terminated_paths_array(num_states, indices, num_indices, indices_offset);
                    break;
                }
                case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_COMPACT_STATES: {
                    int *active_terminated_states = get_pointer<int>(args.values[0]);
                    int active_states_offset = get_scalar<int>(args.values[1]);
                    int terminated_states_offset = get_scalar<int>(args.values[2]);
                    int work_size = get_scalar<int>(args.values[3]);
                    ::simple_integrator_compact_states(active_terminated_states, active_states_offset, terminated_states_offset, work_size);
                    break;
                }
                case DeviceKernel::DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_CHECK: {
                    float *render_buffer = get_pointer<float>(args.values[0]);
                    int sx = get_scalar<int>(args.values[1]);
                    int sy = get_scalar<int>(args.values[2]);
                    int sw = get_scalar<int>(args.values[3]);
                    int sh = get_scalar<int>(args.values[4]);
                    float threshold = get_scalar<float>(args.values[5]);
                    int reset = get_scalar<int>(args.values[6]);
                    int offset = get_scalar<int>(args.values[7]);
                    int stride = get_scalar<int>(args.values[8]);
                    uint *num_active_pixels = get_pointer<uint>(args.values[9]);
                    ::simple_adaptive_sampling_convergence_check(render_buffer, sx, sy, sw, sh, threshold, reset, offset, stride, num_active_pixels);
                    break;
                }
                default:
                    std::cout << "unknown kernel " << device_kernel_as_string(kernel) << std::endl;
                    break;
        }
      }
    }

    *kernel_globals.idx = local_idx;
    *kernel_globals.dim = local_dim;
    *kernel_globals.bid = local_bid;


    debug_enqueue_end();
    return true;
}
bool SimpleDeviceQueue::synchronize() { return true; }
void SimpleDeviceQueue::zero_to_device(device_memory &mem) {device->mem_zero(mem);}
void SimpleDeviceQueue::copy_to_device(device_memory &mem) {device->mem_copy_to(mem);}
void SimpleDeviceQueue::copy_from_device(device_memory &mem) {device->mem_copy_from(mem,0,0,0,0);}
bool SimpleDeviceQueue::supports_local_atomic_sort() const { return false; }

CCL_NAMESPACE_END