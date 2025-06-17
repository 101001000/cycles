/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#ifdef WITH_PORTABLERT

#  include "device/portablert/graphics_interop.h"

#  include "device/portablert/device_impl.h"
#  include "device/portablert/util.h"

CCL_NAMESPACE_BEGIN

PortableRTDeviceGraphicsInterop::PortableRTDeviceGraphicsInterop(PortableRTDeviceQueue *queue)
    : queue_(queue), device_(static_cast<PortableRTDevice *>(queue->device))
{
}

PortableRTDeviceGraphicsInterop::~PortableRTDeviceGraphicsInterop()
{
  PortableRTContextScope scope(device_);

  if (cu_graphics_resource_) {
    cuda_device_assert(device_, cuGraphicsUnregisterResource(cu_graphics_resource_));
  }
}

void PortableRTDeviceGraphicsInterop::set_display_interop(
    const DisplayDriver::GraphicsInterop &display_interop)
{
  const int64_t new_buffer_area = int64_t(display_interop.buffer_width) *
                                  display_interop.buffer_height;

  need_clear_ = display_interop.need_clear;

  if (!display_interop.need_recreate) {
    if (opengl_pbo_id_ == display_interop.opengl_pbo_id && buffer_area_ == new_buffer_area) {
      return;
    }
  }

  PortableRTContextScope scope(device_);

  if (cu_graphics_resource_) {
    cuda_device_assert(device_, cuGraphicsUnregisterResource(cu_graphics_resource_));
  }

  const CUresult result = cuGraphicsGLRegisterBuffer(
      &cu_graphics_resource_, display_interop.opengl_pbo_id, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
  if (result != CUDA_SUCCESS) {
    LOG(ERROR) << "Error registering OpenGL buffer: " << cuewErrorString(result);
  }

  opengl_pbo_id_ = display_interop.opengl_pbo_id;
  buffer_area_ = new_buffer_area;
}

device_ptr PortableRTDeviceGraphicsInterop::map()
{
  if (!cu_graphics_resource_) {
    return 0;
  }

  PortableRTContextScope scope(device_);

  CUdeviceptr cu_buffer;
  size_t bytes;

  cuda_device_assert(device_, cuGraphicsMapResources(1, &cu_graphics_resource_, queue_->stream()));
  cuda_device_assert(
      device_, cuGraphicsResourceGetMappedPointer(&cu_buffer, &bytes, cu_graphics_resource_));

  if (need_clear_) {
    cuda_device_assert(
        device_, cuMemsetD8Async(static_cast<CUdeviceptr>(cu_buffer), 0, bytes, queue_->stream()));

    need_clear_ = false;
  }

  return static_cast<device_ptr>(cu_buffer);
}

void PortableRTDeviceGraphicsInterop::unmap()
{
  PortableRTContextScope scope(device_);

  cuda_device_assert(device_,
                     cuGraphicsUnmapResources(1, &cu_graphics_resource_, queue_->stream()));
}

CCL_NAMESPACE_END

#endif
