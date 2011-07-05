#ifndef CUIMG_LOCAL_JET_STATIC_H_
# define CUIMG_LOCAL_JET_STATIC_H_

# include <cuimg/gpu/image2d.h>
# include <cuimg/gpu/gaussian_static.h>
# include <cuimg/util.h>

namespace cuimg
{
  template <typename TI, typename TO, typename TT, int I, int J, int SIGMA, int KERNEL_HALF_SIZE>
  void local_jet_static(const TI& in, TO& out, TT& tmp, dim3 dimblock = dim3(16, 16))
  {
    assert(in.domain() == out.domain());
    assert(in.domain() == tmp.domain());

    dim3 dimgrid = grid_dimension(in.domain(), dimblock);

    gaussian_static_row2d<typename TI, TT, I, SIGMA, KERNEL_HALF_SIZE>(in, tmp);
    gaussian_static_col2d<typename TT, TO, J, SIGMA, KERNEL_HALF_SIZE>(tmp, out);
    check_cuda_error();
  }

}

#endif
