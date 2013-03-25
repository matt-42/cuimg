#ifndef CUIMG_MEMSET_H_
# define CUIMG_MEMSET_H_

# include <cstring>
# include <cuimg/gpu/device_image2d.h>
# include <cuimg/cpu/host_image2d.h>
# include <cuimg/gpu/device_image3d.h>
# include <cuimg/cpu/host_image3d.h>
# include <cuimg/error.h>

namespace cuimg
{
  template <typename T>
  void memset(device_image2d<T>& out, int v)
  {
    // cudaMemset2D(out.data(), out.pitch(), v,
    //              out.pitch(), out.nrows());
    cudaMemset(out.begin(), v, out.nrows() * out.pitch());
    check_cuda_error();
  }

  template <typename T>
  void memset(host_image2d<T>& out, int v)
  {
    ::memset(out.begin(), v, out.pitch() * out.nrows());
  }

}

#endif
