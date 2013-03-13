#ifndef CUIMG_GAUSSIAN_BLUR_NPP_H_
# define CUIMG_GAUSSIAN_BLUR_NPP_H_

# include <npp.h>
# include <nppi_computer_vision.h>
# include <nppi_filtering_functions.h>
# include <cuimg/improved_builtin.h>
# include <cuimg/obox2d.h>
# include <cuimg/obox3d.h>
# include <cuimg/error.h>
# include <cuimg/gaussian_kernel.h>

namespace cuimg
{
  struct npp_gaussian_kernel
  {
    inline npp_gaussian_kernel(float sigma, int hw)
    {
      half_width_ = hw;
      int size = 2 * half_width_ + 1;
      Npp32s host_kernel[size];
      sum_ = 0;
      for (int i = -half_width_; i <= half_width_; i++)
      {
	host_kernel[i + half_width_] = gaussian_derivative<0>(sigma, i) * 100;
	sum_ += host_kernel[i + half_width_];
      }

      cudaMalloc(&data_, size * sizeof(int));
      cudaMemcpy(data_, host_kernel, size * sizeof(int), cudaMemcpyHostToDevice);
    }

    inline ~npp_gaussian_kernel()
    {
      cudaFree(data_);
    }

    inline Npp32s* kernel() const { return data_; }
    inline int half_width() const { return half_width_; }
    inline float sum() const { return sum_; }

    Npp32s* data_;
    int half_width_;
    float sum_;
  };

  template <typename I>
  void gaussian_blur(const I& src, I& dst, I& tmp, const npp_gaussian_kernel& k)
  {
    int size = 2 * k.half_width() + 1;

    NppiSize roi = {src.ncols() - k.half_width(), src.nrows() - k.half_width()};
    NppStatus s;
    s = nppiFilterColumn_8u_C1R((Npp8u*)src.data(), src.pitch(),
				(Npp8u*)((unsigned long)tmp.data() + k.half_width()*tmp.pitch()),
				tmp.pitch(),
				roi,
				k.kernel(),
				size, size - 1,
				int(k.sum()));
    assert(s == NPP_SUCCESS);
    s = nppiFilterRow_8u_C1R((Npp8u*)tmp.data(), tmp.pitch(),
			     (Npp8u*)((unsigned long)dst.data() + k.half_width()*dst.pitch()),
			     dst.pitch(),
			     roi,
			     k.kernel(),
			     size, size - 1,
			     int(k.sum()));
    assert(s == NPP_SUCCESS);
  }

  template <typename I>
  void gaussian_blur(const I& src, I& dst, I& tmp, float sigma, int half_width)
  {
    npp_gaussian_kernel k(sigma, half_width);
    gaussian_blur(src, dst, tmp, k);
  }

}

#endif
