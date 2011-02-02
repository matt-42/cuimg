#ifndef CUIMG_COPY_H_
# define CUIMG_COPY_H_

# include <cuimg/image2d.h>
# include <cuimg/host_image2d.h>
# include <cuimg/image3d.h>
# include <cuimg/host_image3d.h>
# include <cuimg/error.h>

namespace cuimg
{
  template <typename T>
  void copy(const image2d<T>& in, host_image2d<T>& out)
  {
    assert(in.domain() == out.domain());
    cudaMemcpy2D(out.data(), out.pitch(), in.data(), in.pitch(), in.ncols() * sizeof(T), in.nrows(),
               cudaMemcpyDeviceToHost);
    check_cuda_error();
  }

  template <typename T>
  void copy(const host_image2d<T>& in, image2d<T>& out)
  {
    assert(in.domain() == out.domain());
    cudaMemcpy2D(out.data(), out.pitch(), in.data(), in.pitch(), in.ncols() * sizeof(T), in.nrows(),
               cudaMemcpyHostToDevice);
    check_cuda_error();
  }

  template <typename T>
  void copy(const image3d<T>& in, host_image3d<T>& out)
  {
    assert(in.domain() == out.domain());
    cudaMemcpy3DParms p = {0};
    //memset(&p, 0, sizeof(cudaMemcpy3DParms));
    p.srcPtr = make_cudaPitchedPtr((void*)in.data(), in.pitch(), in.ncols() * sizeof(T), in.nrows());
    p.dstPtr = make_cudaPitchedPtr((void*)out.data(), out.pitch(), out.ncols() * sizeof(T), out.nrows());
    p.extent.width = in.ncols() * sizeof(T);
    p.extent.height = in.nrows();
    p.extent.depth = in.nslices();
    p.kind = cudaMemcpyDeviceToHost;
    cudaMemcpy3D(&p);
    check_cuda_error();
  }

  template <typename T>
  void copy(const host_image3d<T>& in, image3d<T>& out)
  {
    assert(in.domain() == out.domain());
    cudaMemcpy3DParms p = {0};
    //memset(&p, 0, sizeof(cudaMemcpy3DParms));
    p.srcPtr = make_cudaPitchedPtr((void*)in.data(), in.pitch(), in.ncols() * sizeof(T), in.nrows());
    p.dstPtr = make_cudaPitchedPtr((void*)out.data(), out.pitch(), out.ncols() * sizeof(T), out.nrows());
    p.extent.width = in.ncols() * sizeof(T);
    p.extent.height = in.nrows();
    p.extent.depth = in.nslices();
    p.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&p);
    check_cuda_error();
  }

}

#endif
