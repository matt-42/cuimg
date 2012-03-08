#ifndef CUIMG_LOCAL_JET_STATIC_H_
# define CUIMG_LOCAL_JET_STATIC_H_

# include <cuimg/gpu/device_image2d.h>
# include <cuimg/gpu/meta_convolve.h>
# include <cuimg/gpu/meta_convolve2.h>
# include <cuimg/meta_gaussian/meta_gaussian.h>
# include <cuimg/util.h>

namespace cuimg
{
  template <typename TI, typename TO, typename TT, int I, int J, int SIGMA, int KERNEL_HALF_SIZE>
  void local_jet_static(const TI& in, TO& out, TT& tmp,
                        cudaStream_t stream = 0, dim3 dimblock = dim3(16, 16))
  {
    assert(in.domain() == out.domain());
    assert(in.domain() == tmp.domain());

    dim3 dimgrid = grid_dimension(in.domain(), dimblock);

    meta_convolve_row2d<TI, TT, meta_gaussian<I, SIGMA>, KERNEL_HALF_SIZE>
      (in, tmp, stream, dimblock);
    meta_convolve_col2d<TT, TO, meta_gaussian<J, SIGMA>, KERNEL_HALF_SIZE>
      (tmp, out, stream, dimblock);
  }

  template <typename TI, typename TO, typename TT,
            int I1, int J1, int SIGMA1,
            int I2, int J2, int SIGMA2,
            int KERNEL_HALF_SIZE>
  void local_jet_static(const TI& in, TO& out1, TO& out2, TT& tmp1, TT& tmp2,
                        cudaStream_t stream = 0, dim3 dimblock = dim3(16, 16))
  {
    assert(in.domain() == out1.domain());
    assert(in.domain() == tmp1.domain());
    assert(in.domain() == out2.domain());
    assert(in.domain() == tmp2.domain());

    dim3 dimgrid = grid_dimension(in.domain(), dimblock);

    meta_convolve_row2d<TI, TT, meta_gaussian<I1, SIGMA1>, meta_gaussian<I2, SIGMA2>,
      KERNEL_HALF_SIZE>
      (in, tmp1, tmp2, stream, dimblock);
    meta_convolve_col2d<TT, TO, meta_gaussian<J1, SIGMA1>, meta_gaussian<J2, SIGMA2>,
      KERNEL_HALF_SIZE>
      (tmp1, tmp2, out1, out2, stream, dimblock);
  }

  template <int I1, int J1, int SIGMA1, int I2, int J2, int SIGMA2, int KERNEL_HALF_SIZE>
  struct local_jet_static2_
  {
    template <typename TI, typename TO, typename TT>
    static void run(const TI& in, TO& out1, TO& out2, TT& tmp1, TT& tmp2,
                        cudaStream_t stream = 0, dim3 dimblock = dim3(16, 16))
    {
      local_jet_static<TI, TO, TT, I1, J1, SIGMA1, I2, J2, SIGMA2, KERNEL_HALF_SIZE>
        (in, out1, out2, tmp1, tmp2, stream, dimblock);
    }
  };

  template <int I, int J, int SIGMA, int KERNEL_HALF_SIZE>
  struct local_jet_static_
  {
    template <typename TI, typename TO, typename TT>
    static void run(const TI& in, TO& out, TT& tmp,
                        cudaStream_t stream = 0, dim3 dimblock = dim3(16, 16))
    {
      local_jet_static<TI, TO, TT, I, J, SIGMA, KERNEL_HALF_SIZE>
        (in, out, tmp, stream, dimblock);
    }
  };
}

#endif
