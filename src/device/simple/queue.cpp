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

extern void simple_integrator_intersect_closest2(const int *path_index_array,
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

extern void simple_integrator_shade_light(const int *path_index_array, float *render_buffer, const int work_size);

extern void simple_adaptive_sampling_filter_y(float *render_buffer, const int sx, const int sy, const int sw, const int sh, const int offset, const int stride);
extern void simple_adaptive_sampling_filter_x(float *render_buffer, const int sx, const int sy, const int sw, const int sh, const int offset, const int stride);

CCL_NAMESPACE_BEGIN


SimpleDeviceQueue::SimpleDeviceQueue(SimpleDevice *device) : DeviceQueue(device), device(device) {}
SimpleDeviceQueue::~SimpleDeviceQueue() {}

int SimpleDeviceQueue::num_concurrent_states(const size_t state_size) const { return 1048576; }
int SimpleDeviceQueue::num_concurrent_busy_states(const size_t state_size) const { return 64; }
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

   // std::cout << "Enqueueing kernel " << kernel << " with work size " << work_size << " and args size " << args.count << std::endl;

    debug_enqueue_begin(kernel, work_size); 

    std::vector<prt::Ray> dummy_rays(work_size);
    std::vector<unsigned char> dummy_output(work_size);

    switch(kernel) {

    case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST: {

        struct KernelArgs {
            int *path_index_array;
            float *render_buffer;
            int work_size;
        };

        assert(args.count == 3);
        
        KernelArgs ka;
        ka.path_index_array = get_pointer<int>(args.values[0]);
        ka.render_buffer = get_pointer<float>(args.values[1]);
        ka.work_size = get_scalar<int>(args.values[2]);
        device->m_backend->parallel_invoke("simple_integrator_intersect_closest", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }

    case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA: {
        struct KernelArgs {
            ccl::KernelWorkTile *tiles;
            int num_tiles;
            float *render_buffer;
            int max_tile_work;
        };

        assert(args.count == 4);

        KernelArgs ka;
        ka.tiles = get_pointer<ccl::KernelWorkTile>(args.values[0]);
        ka.num_tiles = get_scalar<int>(args.values[1]);
        ka.render_buffer = get_pointer<float>(args.values[2]);
        ka.max_tile_work = get_scalar<int>(args.values[3]);
        device->m_backend->parallel_invoke("simple_integrator_init_from_camera", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }

    case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_RESET: {
        struct KernelArgs {
            int num_states;
        };

        assert(args.count == 1);
        
        KernelArgs ka;
        ka.num_states = get_scalar<int>(args.values[0]);
        std::cout << "Invocando simple_integrator_reset" <<  ka.num_states  << std::endl;
        device->m_backend->parallel_invoke("simple_integrator_reset", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }

    case DeviceKernel::DEVICE_KERNEL_PREFIX_SUM: {
        struct KernelArgs {
            int *counter;
            int *prefix_sum;
            int num_values;
        };

        assert(args.count == 3);

        KernelArgs ka;
        ka.counter = get_pointer<int>(args.values[0]);
        ka.prefix_sum = get_pointer<int>(args.values[1]);
        ka.num_values = get_scalar<int>(args.values[2]);
        device->m_backend->parallel_invoke("simple_prefix_sum", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }

    case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_SORTED_PATHS_ARRAY: {
        struct KernelArgs {
            int num_states;
            int num_states_limit;
            int *indices;
            int *num_indices;
            int *key_counter;
            int *key_prefix_sum;
            int kernel_index;
        };

        assert(args.count == 7);
        
        KernelArgs ka;
        ka.num_states = get_scalar<int>(args.values[0]);
        ka.num_states_limit = get_scalar<int>(args.values[1]);
        ka.indices = get_pointer<int>(args.values[2]);
        ka.num_indices = get_pointer<int>(args.values[3]);
        ka.key_counter = get_pointer<int>(args.values[4]);
        ka.key_prefix_sum = get_pointer<int>(args.values[5]);
        ka.kernel_index = get_scalar<int>(args.values[6]);
        device->m_backend->parallel_invoke("simple_integrator_sorted_paths_array", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }

    case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE: {
        struct KernelArgs {
            int *path_index_array;
            float *render_buffer;
            int work_size;
        };

        assert(args.count == 3);
        
        KernelArgs ka;
        ka.path_index_array = get_pointer<int>(args.values[0]);
        ka.render_buffer = get_pointer<float>(args.values[1]);
        ka.work_size = get_scalar<int>(args.values[2]);
        device->m_backend->parallel_invoke("simple_integrator_shade_surface", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }

    case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND: {
        struct KernelArgs {
            int *path_index_array;
            float *render_buffer;
            int work_size;
        };

        assert(args.count == 3);

        KernelArgs ka;
        ka.path_index_array = get_pointer<int>(args.values[0]);
        ka.render_buffer = get_pointer<float>(args.values[1]);
        ka.work_size = get_scalar<int>(args.values[2]);
        device->m_backend->parallel_invoke("simple_integrator_shade_background", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }

    case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_QUEUED_PATHS_ARRAY: {
        struct KernelArgs {
            int num_states;
            int *indices;
            int *num_indices;
            int kernel_index;
        };

        assert(args.count == 4);

        KernelArgs ka;
        ka.num_states = get_scalar<int>(args.values[0]);
        ka.indices = get_pointer<int>(args.values[1]);
        ka.num_indices = get_pointer<int>(args.values[2]);
        ka.kernel_index = get_scalar<int>(args.values[3]);
        device->m_backend->parallel_invoke("simple_integrator_queued_paths_array", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }

    case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_COMPACT_PATHS_ARRAY: {
        struct KernelArgs {
            int num_states;
            int *indices;
            int *num_indices;
            int num_active_paths;
        };

        assert(args.count == 4);

        KernelArgs ka;
        ka.num_states = get_scalar<int>(args.values[0]);
        ka.indices = get_pointer<int>(args.values[1]);
        ka.num_indices = get_pointer<int>(args.values[2]);
        ka.num_active_paths = get_scalar<int>(args.values[3]);
        device->m_backend->parallel_invoke("simple_integrator_compact_paths_array", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }

    case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_TERMINATED_PATHS_ARRAY: {
        struct KernelArgs {
            int num_states;
            int *indices;
            int *num_indices;
            int indices_offset;
        };

        assert(args.count == 4);

        KernelArgs ka;
        ka.num_states = get_scalar<int>(args.values[0]);
        ka.indices = get_pointer<int>(args.values[1]);
        ka.num_indices = get_pointer<int>(args.values[2]);
        ka.indices_offset = get_scalar<int>(args.values[3]);
        device->m_backend->parallel_invoke("simple_integrator_terminated_paths_array", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }

    case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_COMPACT_STATES: {
        struct KernelArgs {
            int *active_terminated_states;
            int active_states_offset;
            int terminated_states_offset;
            int work_size;
        };

        assert(args.count == 4);

        KernelArgs ka;
        ka.active_terminated_states = get_pointer<int>(args.values[0]);
        ka.active_states_offset = get_scalar<int>(args.values[1]);
        ka.terminated_states_offset = get_scalar<int>(args.values[2]);
        ka.work_size = get_scalar<int>(args.values[3]);
        device->m_backend->parallel_invoke("simple_integrator_compact_states", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }

    case DeviceKernel::DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_CHECK: {
        struct KernelArgs {
            float *render_buffer;
            int sx, sy, sw, sh;
            float threshold;
            int reset;
            int offset;
            int stride;
            uint *num_active_pixels;
        };

        assert(args.count == 9);

        KernelArgs ka;
        ka.render_buffer = get_pointer<float>(args.values[0]);
        ka.sx = get_scalar<int>(args.values[1]);
        ka.sy = get_scalar<int>(args.values[2]);
        ka.sw = get_scalar<int>(args.values[3]);
        ka.sh = get_scalar<int>(args.values[4]);
        ka.threshold = get_scalar<float>(args.values[5]);
        ka.reset = get_scalar<int>(args.values[6]);
        ka.offset = get_scalar<int>(args.values[7]);
        ka.stride = get_scalar<int>(args.values[8]);
        ka.num_active_pixels = get_pointer<uint>(args.values[9]);
        device->m_backend->parallel_invoke("simple_adaptive_sampling_convergence_check", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }

    case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT: {
        struct KernelArgs {
            int *path_index_array;
            float *render_buffer;
            int work_size;
        };

        assert(args.count == 3);

        KernelArgs ka;
        ka.path_index_array = get_pointer<int>(args.values[0]);
        ka.render_buffer = get_pointer<float>(args.values[1]);
        ka.work_size = get_scalar<int>(args.values[2]);
        device->m_backend->parallel_invoke("simple_integrator_shade_light", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }

    case DeviceKernel::DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_FILTER_X: {
        struct KernelArgs {
            float *render_buffer;
            int sx, sy, sw, sh;
            int offset;
            int stride;
        };

        assert(args.count == 6);

        KernelArgs ka;
        ka.render_buffer = get_pointer<float>(args.values[0]);
        ka.sx = get_scalar<int>(args.values[1]);
        ka.sy = get_scalar<int>(args.values[2]);
        ka.sw = get_scalar<int>(args.values[3]);
        ka.sh = get_scalar<int>(args.values[4]);
        ka.offset = get_scalar<int>(args.values[5]);
        ka.stride = get_scalar<int>(args.values[6]);
        device->m_backend->parallel_invoke("simple_adaptive_sampling_filter_x", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }

    case DeviceKernel::DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_FILTER_Y: {
        struct KernelArgs {
            float *render_buffer;
            int sx, sy, sw, sh;
            int offset;
            int stride;
        };

        assert(args.count == 6);

        KernelArgs ka;
        ka.render_buffer = get_pointer<float>(args.values[0]);
        ka.sx = get_scalar<int>(args.values[1]);
        ka.sy = get_scalar<int>(args.values[2]);
        ka.sw = get_scalar<int>(args.values[3]);
        ka.sh = get_scalar<int>(args.values[4]);
        ka.offset = get_scalar<int>(args.values[5]);
        ka.stride = get_scalar<int>(args.values[6]);
        device->m_backend->parallel_invoke("simple_adaptive_sampling_filter_y", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }
    default:
        std::cout << "unknown kernel " << device_kernel_as_string(kernel) << std::endl;
        break;
    }



    /*
    for(int i = 0; i < work_size; i++){
    
        *kernel_globals.idx = 0;
        *kernel_globals.dim = 1;
        *kernel_globals.bid = i;
        
        switch(kernel) {

            case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST: {
                break;
            }

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
            case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT: {
                int *path_index_array = get_pointer<int>(args.values[0]);
                float *render_buffer = get_pointer<float>(args.values[1]);
                ::simple_integrator_shade_light(path_index_array, render_buffer, work_size);
                break;
            }
            case DeviceKernel::DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_FILTER_X: {
                float *render_buffer = get_pointer<float>(args.values[0]);
                int sx = get_scalar<int>(args.values[1]);
                int sy = get_scalar<int>(args.values[2]);
                int sw = get_scalar<int>(args.values[3]);
                int sh = get_scalar<int>(args.values[4]);
                int offset = get_scalar<int>(args.values[5]);
                int stride = get_scalar<int>(args.values[6]);
                ::simple_adaptive_sampling_filter_x(render_buffer, sx, sy, sw, sh, offset, stride);
                break;
            }
            case DeviceKernel::DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_FILTER_Y: {
                float *render_buffer = get_pointer<float>(args.values[0]);
                int sx = get_scalar<int>(args.values[1]);
                int sy = get_scalar<int>(args.values[2]);
                int sw = get_scalar<int>(args.values[3]);
                int sh = get_scalar<int>(args.values[4]);
                int offset = get_scalar<int>(args.values[5]);
                int stride = get_scalar<int>(args.values[6]);
                ::simple_adaptive_sampling_filter_y(render_buffer, sx, sy, sw, sh, offset, stride);
                break;
            }
            default:
                std::cout << "unknown kernel " << device_kernel_as_string(kernel) << std::endl;
                break;
            }
    }*/


    debug_enqueue_end();
    return true;
}
bool SimpleDeviceQueue::synchronize() { return true; }
void SimpleDeviceQueue::zero_to_device(device_memory &mem) {device->mem_zero(mem);}
void SimpleDeviceQueue::copy_to_device(device_memory &mem) {device->mem_copy_to(mem);}
void SimpleDeviceQueue::copy_from_device(device_memory &mem) {device->mem_copy_from(mem,0,0,0,0);}
bool SimpleDeviceQueue::supports_local_atomic_sort() const { return false; }

CCL_NAMESPACE_END