/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#ifdef WITH_PORTABLERT

#  include "device/memory.h"
#  include "device/queue.h"

#  include "device/portablert/util.h"

CCL_NAMESPACE_BEGIN

class PortableRTDevice;
class device_memory;

/* Base class for CUDA queues. */
class PortableRTDeviceQueue : public DeviceQueue {
 public:
  PortableRTDeviceQueue(PortableRTDevice *device);
  ~PortableRTDeviceQueue() override;

  int num_concurrent_states(const size_t state_size) const override;
  int num_concurrent_busy_states(const size_t state_size) const override;

  void init_execution() override;

  bool enqueue(DeviceKernel kernel,
               const int work_size,
               const DeviceKernelArguments &args) override;

  bool synchronize() override;

  void zero_to_device(device_memory &mem) override;
  void copy_to_device(device_memory &mem) override;
  void copy_from_device(device_memory &mem) override;

  virtual CUstream stream()
  {
    return cuda_stream_;
  }

  unique_ptr<DeviceGraphicsInterop> graphics_interop_create() override;

 protected:
  PortableRTDevice *cuda_device_;
  CUstream cuda_stream_;

  void assert_success(CUresult result, const char *operation);
};

CCL_NAMESPACE_END

#endif /* WITH_CUDA */
