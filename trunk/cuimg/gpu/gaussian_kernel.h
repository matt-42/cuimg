#ifndef CUIMG_GAUSSIAN_KERNEL_H_
# define CUIMG_GAUSSIAN_KERNEL_H_

# include <cuimg/gpu/device_image2d.h>
# include <cuimg/gaussian_kernel.h>
# include <cuimg/util.h>

namespace cuimg
{
  // Fixme missing implementations.

  template <typename U, typename V, typename WW>
  void gaussian_kernelRows(const device_image2d<U>& in, device_image2d<V>& out, const WW& weighted_window);

  template <typename U, typename V, typename WW>
  void gaussian_kernelCols(const device_image2d<U>& in, device_image2d<V>& out, const WW& weighted_window);

}

#endif
