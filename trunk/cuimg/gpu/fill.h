#ifndef CUIMG_FILL_H_
# define CUIMG_FILL_H_

# include <cuimg/improved_builtin.h>
# include <cuimg/gpu/image2d.h>
# include <cuimg/error.h>
# include <cuimg/util.h>

namespace cuimg
{

  namespace internal
  {
#ifdef NVCC
    template<typename U>
    __global__ void fill_kernel(kernel_image2d<U> out, U v)
    {
      point2d<int> p = thread_pos2d();
      if (out.has(p))
        out(p) = v;
    }
#endif
  }


  template<typename U>
  void fill(image2d<U>& out,
            const U& v,
            dim3 dim_block = dim3(16, 16, 1))
  {
    dim3 dim_grid = grid_dimension(out.domain(), dim_block);
#ifdef NVCC
    internal::fill_kernel<<<dim_grid, dim_block>>>(mki(out), v);
#endif
  }


}

#endif
