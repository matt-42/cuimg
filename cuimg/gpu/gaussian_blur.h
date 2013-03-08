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

  template <typename I>
  void gaussian_blur(const I& src, I& dst, I& tmp, float sigma, int half_width)
  {
    int size = 2 * half_width + 1;

    Npp32s* host_kernel = new int[size];
    float sum = 0;
    for (int i = -half_width; i < half_width; i++)
    {
      host_kernel[i + half_width] = gaussian_derivative<0>(sigma, i) * 100;
      sum += host_kernel[i + half_width];
    }

    Npp32s* pKernel;
    cudaMalloc(&pKernel, size * sizeof(int));
    cudaMemcpy(pKernel, host_kernel, size * sizeof(int), cudaMemcpyHostToDevice);

    NppiSize roi = {src.ncols() - half_width, src.nrows() - half_width};
    NppStatus s;
    s = nppiFilterColumn_8u_C1R((Npp8u*)src.data(), src.pitch(),
				(Npp8u*)((unsigned long)tmp.data() + (half_width)*tmp.pitch()), 
				tmp.pitch(),
				roi,
				pKernel,
				size, size - 1,
				int(sum));
    assert(s == NPP_SUCCESS);
    s = nppiFilterRow_8u_C1R((Npp8u*)tmp.data(), tmp.pitch(),
				(Npp8u*)((unsigned long)dst.data() + half_width), 
				dst.pitch(),
				roi,
				pKernel,
				size, size - 1,
				int(sum));
    assert(s == NPP_SUCCESS);

    delete[] host_kernel;
    cudaFree(pKernel);
  }

}

#endif
