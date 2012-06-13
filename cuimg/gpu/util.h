#ifndef CUIMG_GPU_UTIL_H_
# define CUIMG_GPU_UTIL_H_

# include <cuda_runtime.h>
# include <cuda.h>
# include <cuimg/util.h>

# include <cuimg/gpu/device_image3d.h>

namespace cuimg
{

  template <typename T>
  void prepare_texture_3d(cudaArray** a, texture<T, 3, cudaReadModeElementType>& tex,
                          unsigned nslices, unsigned nrows, unsigned ncols)
  {
    cudaChannelFormatDesc ca_descriptor;
    cudaExtent ca_extent;
    ca_descriptor = cudaCreateChannelDesc<T>();
    ca_extent.width  = ncols;
    ca_extent.height = nrows;
    ca_extent.depth  = nslices;
    cudaMalloc3DArray(a, &ca_descriptor, ca_extent);
    cudaBindTextureToArray(tex, *a, ca_descriptor);
    check_cuda_error();
  }

  template <typename T>
  void copy_nslices_to_array(device_image3d<T>& in, cudaArray* arr, unsigned start_slice,
                             unsigned n_slices)
  {
    cudaExtent ca_extent;
    ca_extent.width  = in.ncols();
    ca_extent.height = in.nrows();
    ca_extent.depth  = n_slices;

    cudaMemcpy3DParms cpy_params = {0};
    cpy_params.extent   = ca_extent;
    cpy_params.kind     = cudaMemcpyDeviceToDevice;
    cpy_params.dstArray = arr;
    cpy_params.srcPtr   = make_cudaPitchedPtr((void*)in.data(), in.pitch(), in.ncols() * sizeof(T), in.nrows());

    cpy_params.srcPos   = make_cudaPos(0, 0, start_slice);
    cpy_params.dstPos   = make_cudaPos(0, 0, start_slice);

    cudaMemcpy3D( &cpy_params );
    check_cuda_error();
  }

}

#endif
