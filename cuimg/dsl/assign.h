#ifndef CUIMG_GPU_ASSIGN_H_
# define CUIMG_GPU_ASSIGN_H_


# include <cuimg/gpu/device_image2d.h>
# include <cuimg/gpu/kernel_image2d.h>
# include <cuimg/util.h>
# include <cuimg/gpu/util.h>

namespace cuimg
{
  kernel_image2d<int> coucou;

    template <typename I>
      __global__ void test(I out)
    {
    }


}

#endif
