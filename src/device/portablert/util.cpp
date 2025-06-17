/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#ifdef WITH_PORTABLERT

#  include "device/portablert/util.h"
#  include "device/portablert/device_impl.h"

CCL_NAMESPACE_BEGIN

PortableRTContextScope::PortableRTContextScope(PortableRTDevice *device) : device(device)
{
  cuda_device_assert(device, cuCtxPushCurrent(device->cuContext));
}

PortableRTContextScope::~PortableRTContextScope()
{
  cuda_device_assert(device, cuCtxPopCurrent(nullptr));
}

#  ifndef WITH_CUDA_DYNLOAD
const char *cuewErrorString(CUresult result)
{
  /* We can only give error code here without major code duplication, that
   * should be enough since dynamic loading is only being disabled by folks
   * who knows what they're doing anyway.
   *
   * NOTE: Avoid call from several threads.
   */
  static string error;
  error = string_printf("%d", result);
  return error.c_str();
}

const char *cuewCompilerPath()
{
  return CYCLES_CUDA_NVCC_EXECUTABLE;
}

int cuewCompilerVersion()
{
  return (CUDA_VERSION / 100) + (CUDA_VERSION % 100 / 10);
}
#  endif

CCL_NAMESPACE_END

#endif /* WITH_CUDA */
