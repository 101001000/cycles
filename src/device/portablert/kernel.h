/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#ifdef WITH_PORTABLERT

#  include "device/kernel.h"

#  ifdef WITH_CUDA_DYNLOAD
#    include "cuew.h"
#  else
#    include <cuda.h>
#  endif

CCL_NAMESPACE_BEGIN

class PortableRTDevice;

/* CUDA kernel and associate occupancy information. */
class PortableRTDeviceKernel {
 public:
  CUfunction function = nullptr;

  int num_threads_per_block = 0;
  int min_blocks = 0;
};

/* Cache of CUDA kernels for each DeviceKernel. */
class PortableRTDeviceKernels {
 public:
  void load(PortableRTDevice *device);
  const PortableRTDeviceKernel &get(DeviceKernel kernel) const;
  bool available(DeviceKernel kernel) const;

 protected:
  PortableRTDeviceKernel kernels_[DEVICE_KERNEL_NUM];
  bool loaded = false;
};

CCL_NAMESPACE_END

#endif /* WITH_CUDA */
