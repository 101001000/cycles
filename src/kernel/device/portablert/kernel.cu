/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* CUDA kernel entry points */

#ifdef __CUDA_ARCH__

#  include "kernel/device/portablert/compat.h"
#  include "kernel/device/portablert/config.h"
#  include "kernel/device/portablert/globals.h"

#  include "kernel/device/gpu/image.h"
#  include "kernel/device/gpu/kernel.h"

#endif
