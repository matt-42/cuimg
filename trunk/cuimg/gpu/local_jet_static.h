#ifndef CUIMG_LOCAL_JET_STATIC_H_
# define CUIMG_LOCAL_JET_STATIC_H_

# include <cuimg/gpu/image2d.h>
# include <cuimg/gpu/meta_convolve.h>
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
    check_cuda_error();
  }

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
