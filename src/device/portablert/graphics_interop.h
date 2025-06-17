/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#ifdef WITH_PORTABLERT

#  include "device/graphics_interop.h"

#  ifdef WITH_CUDA_DYNLOAD
#    include "cuew.h"
#  else
#    include <cuda.h>
#  endif

CCL_NAMESPACE_BEGIN

class PortableRTDevice;
class PortableRTDeviceQueue;

class PortableRTDeviceGraphicsInterop : public DeviceGraphicsInterop {
 public:
  explicit PortableRTDeviceGraphicsInterop(PortableRTDeviceQueue *queue);

  PortableRTDeviceGraphicsInterop(const PortableRTDeviceGraphicsInterop &other) = delete;
  PortableRTDeviceGraphicsInterop(PortableRTDeviceGraphicsInterop &&other) noexcept = delete;

  ~PortableRTDeviceGraphicsInterop() override;

  PortableRTDeviceGraphicsInterop &operator=(const PortableRTDeviceGraphicsInterop &other) = delete;
  PortableRTDeviceGraphicsInterop &operator=(PortableRTDeviceGraphicsInterop &&other) = delete;

  void set_display_interop(const DisplayDriver::GraphicsInterop &display_interop) override;

  device_ptr map() override;
  void unmap() override;

 protected:
  PortableRTDeviceQueue *queue_ = nullptr;
  PortableRTDevice *device_ = nullptr;

  /* OpenGL PBO which is currently registered as the destination for the CUDA buffer. */
  int64_t opengl_pbo_id_ = 0;
  /* Buffer area in pixels of the corresponding PBO. */
  int64_t buffer_area_ = 0;

  /* The destination was requested to be cleared. */
  bool need_clear_ = false;

  CUgraphicsResource cu_graphics_resource_ = nullptr;
};

CCL_NAMESPACE_END

#endif
