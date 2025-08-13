#pragma once

#include <cstdint>
#include <math.h>
#include <cassert>
#include <climits>
#include <cstring>
#include <iostream>

#define __KERNEL_GPU__
#define __KERNEL_SIMPLE__
#define CCL_NAMESPACE_BEGIN
#define CCL_NAMESPACE_END

#ifndef ATTR_FALLTHROUGH
#  define ATTR_FALLTHROUGH
#endif


#define ccl_gpu_kernel_signature(name, ...) simple_##name(__VA_ARGS__)
#define ccl_gpu_kernel_postfix
#define ccl_gpu_kernel(block_num_threads, thread_num_registers) \
  void 

#define ccl_gpu_kernel_threads(block_num_threads) \
  void
#define ccl_device
#define ccl_gpu_shared
#define ccl_gpu_block_dim_x (*kernel_globals.dim)
#define ccl_gpu_thread_idx_x (*kernel_globals.idx)
#define ccl_gpu_warp_size 1
#define ccl_gpu_block_idx_x (*kernel_globals.bid)
#define ccl_gpu_syncthreads void
#define ccl_gpu_ballot(predicate) (predicate ? 1 : 0)
#define ccl_gpu_kernel_call(x) x
#define ccl_gpu_thread_mask(thread_warp) \
    ((thread_warp) >= 1 ? 1 : 0)  // Máscara para 1 hilo
#define ccl_gpu_global_id_x() (ccl_gpu_block_idx_x * ccl_gpu_block_dim_x + \
  ccl_gpu_thread_idx_x)
#define kernel_data kernel_globals.__data
#define kernel_integrator_state kernel_globals.integrator_state
#define ccl_device_inline inline
#define ccl_device_inline_method ccl_device
#define ccl_private
#define ccl_try_align(...) __attribute__((aligned(__VA_ARGS__)))
#define ccl_device_template_spec template<> ccl_device_inline
#define ccl_device_forceinline static inline
#define ccl_global
#define ccl_align(n) __attribute__((aligned(n)))
#define ccl_restrict __restrict__
#define ccl_static_constexpr static constexpr
#define ccl_device_noinline __attribute__((noinline))
#define ccl_inline_constant const constexpr
#define ccl_constant const
#define kernel_assert(cond)
#define ccl_ray_data ccl_private
#define ccl_gpu_kernel_within_bounds(i, n) ((i) < (n))
#define ccl_optional_struct_init
#define ccl_device_noinline_cpu ccl_device
#define __device__

#define ccl_gpu_kernel_lambda(func, ...) \
  struct KernelLambda { \
    __VA_ARGS__; \
    int operator()(const int state) \
    { \
      return (func); \
    } \
  } ccl_gpu_kernel_lambda_pass


struct CPUTexture2D {
  const void *pixels;    /* pointer a la imagen en memoria lineal   */
  int         width;
  int         height;
  int         channels;  /* 1, 3, 4 …                                */
  /* flags de interpolación, extensión, etc. si los necesitas        */
};

struct CPUTexture3D {
  const void *voxels;
  int         width, height, depth;
  int         channels;
};

/* Alias que sustituye al handle GPU.  A efectos prácticos es un puntero. */
using ccl_gpu_tex_object_2D = const CPUTexture2D *;
using ccl_gpu_tex_object_3D = const CPUTexture3D *;

ccl_device_inline int clampi(int x, int lo, int hi)
{
  return (x < lo) ? lo : (x > hi ? hi : x);
}

template<typename T>
ccl_device_forceinline T ccl_gpu_tex_object_read_2D(const ccl_gpu_tex_object_2D texobj,
                                                    const float fx, const float fy)
{
  std::cout << "reading 2D text "  << std::endl;
  const CPUTexture2D &tex = *texobj;

  /* Coordenadas normalizadas [0..1] → texel */
  const float u = fx * tex.width  - 0.5f;
  const float v = fy * tex.height - 0.5f;

  const int ix = clampi(int(std::floor(u + 0.5f)), 0, tex.width  - 1);
  const int iy = clampi(int(std::floor(v + 0.5f)), 0, tex.height - 1);

  const size_t idx = (size_t)iy * tex.width + ix;
  const T *ptr = reinterpret_cast<const T *>(tex.pixels);

  return ptr[idx];
}

template<typename T>
ccl_device_forceinline T ccl_gpu_tex_object_read_3D(const ccl_gpu_tex_object_3D texobj,
                                                    const float fx, const float fy, const float fz)
{
  std::cout << "reading 3D text "  << std::endl;
  const CPUTexture3D &tex = *texobj;

  const float u = fx * tex.width  - 0.5f;
  const float v = fy * tex.height - 0.5f;
  const float w = fz * tex.depth  - 0.5f;

  const int ix = clampi(int(std::floor(u + 0.5f)), 0, tex.width  - 1);
  const int iy = clampi(int(std::floor(v + 0.5f)), 0, tex.height - 1);
  const int iz = clampi(int(std::floor(w + 0.5f)), 0, tex.depth  - 1);

  const size_t idx = ((size_t)iz * tex.height + iy) * tex.width + ix;
  const T *ptr = reinterpret_cast<const T *>(tex.voxels);

  return ptr[idx];
}

#define atomic_fetch_and_add_uint32(ptr, val) \
  ( ( *(ptr) += (val) ), (*(ptr) - (val)) )

#define atomic_fetch_and_sub_uint32(ptr, val) \
  ( ( *(ptr) -= (val) ), (*(ptr) + (val)) )
#define atomic_add_and_fetch_float(x, y) (*(x) += (y))
#define atomic_compare_and_swap_float(dst, oldval, newval) \
  ((*(dst) == (oldval)) ? (*(dst) = (newval), true) : false)

#include "util/half.h"
#include "util/types.h"

