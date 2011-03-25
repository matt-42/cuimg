#ifndef CUIMG_LOCAL_JET_STATIC_H_
# define CUIMG_LOCAL_JET_STATIC_H_

# include <cuimg/gpu/image2d.h>
# include <cuimg/gpu/gaussian_static.h>
# include <cuimg/util.h>

namespace cuimg
{
  template <typename TI, typename TO, int I, int J, int SIGMA, int KERNEL_HALF_SIZE>
  void local_jet_static(const image2d<TI>& in, image2d<TO>& out, image2d<TO>& tmp, dim3 dimblock = dim3(16, 16))
  {
    assert(in.domain() == out.domain());
    dim3 dimgrid = grid_dimension(in.domain(), dimblock);

    gaussian_static_row2d<typename TI, TO, I, SIGMA, KERNEL_HALF_SIZE>(in, tmp);
    gaussian_static_col2d<typename TI, TO, J, SIGMA, KERNEL_HALF_SIZE>(tmp, out);
  }

}

#endif
