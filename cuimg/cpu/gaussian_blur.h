#ifndef CUIMG_CPU_GAUSSIAN_BLUR_H_
# define CUIMG_CPU_GAUSSIAN_BLUR_H_

# include <cuimg/cpu/host_image2d.h>

namespace cuimg
{

  template <typename T>
  void gaussian_blur_rows_sigma1(const host_image2d<T> in, host_image2d<T> out)
  {
#pragma omp parallel for schedule(static, 2)
    for (unsigned r = 0; r < in.domain().nrows(); r++)
    {
      const T* ptr = &in(r, 0);
      T* out_ptr = &out(r, 0);
      for (int c = 1; c < in.domain().ncols() - 1; c++)
      {
	out_ptr[c] = (2 * ptr[c] + ptr[c - 1] + ptr[c + 1]) / 4;
      }
    }

  }

  template <typename T>
  void gaussian_blur_cols_sigma1(const host_image2d<T> in, host_image2d<T> out)
  {
    int pitch = in.pitch() / sizeof(T);
#pragma omp parallel for schedule(static, 2)
    for (int r = 1; r < in.domain().nrows() - 1; r++)
    {
      const T* ptr = &in(r, 0);
      T* out_ptr = &out(r, 0);
      for (int c = 0; c < in.domain().ncols(); c++)
      {
	out_ptr[c] = (2 * ptr[c] + ptr[c - pitch] + ptr[c + pitch]) / 4;
      }
    }

  }

  template <typename T>
  void gaussian_blur_sigma1(const host_image2d<T> in, host_image2d<T> out, host_image2d<T> tmp)
  {
    gaussian_blur_rows_sigma1(in, tmp);
    gaussian_blur_cols_sigma1(tmp, out);
  }

}

#endif
