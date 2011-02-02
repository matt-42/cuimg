#ifndef CUIMG_LOCAL_JET_STATIC_H_
# define CUIMG_LOCAL_JET_STATIC_H_

# include <cuimg/image2d.h>
# include <cuimg/gaussian_static.h>
# include <cuimg/util.h>

namespace cuimg
{
  template <typename IN, typename OUT, int I, int J, int SIGMA, int KERNEL_HALF_SIZE>
  void local_jet_static(const image2d<IN>& in, image2d<OUT>& out, image2d<OUT>& tmp, dim3 dimblock = dim3(16, 16))
  {
    assert(in.domain() == out.domain());
    dim3 dimgrid = grid_dimension(in.domain(), dimblock);

    gaussian_static_row2d<typename IN, OUT, I, SIGMA, KERNEL_HALF_SIZE>(in, tmp);
    gaussian_static_col2d<typename IN, OUT, J, SIGMA, KERNEL_HALF_SIZE>(tmp, out);
  }

}

#endif
