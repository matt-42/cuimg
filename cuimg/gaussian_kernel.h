#ifndef CUIMG_GAUSSIAN_KERNEL_H_
# define CUIMG_GAUSSIAN_KERNEL_H_

# include <vector>
# include <cmath>
# include <cuimg/util.h>

namespace cuimg
{
  template <int N>
  inline float gaussian_derivative(float sigma, int x);

  template <>
  inline float gaussian_derivative<0>(float sigma, int x)
  {
    return (1.f / (sqrtf(2.f * CUIMG_PI) * sigma))
      * expf(-(x * x) / (2.f * sigma * sigma));
  }

  template <>
  inline float gaussian_derivative<1>(float sigma, int x)
  {
    return -(x / (sqrtf(2.f * CUIMG_PI) * std::pow(sigma, 3.f)))
      * expf(-(x * x) / (2.f * sigma * sigma));
  }

  template <>
  inline float gaussian_derivative<2>(float sigma, int x)
  {
    return (((x - sigma) * (x + sigma)) /
             (sqrtf(2 * CUIMG_PI) * std::pow(sigma, 5.f)))
      * expf(-(x * x) / (2.f * sigma * sigma));
  }

  template <>
  inline float gaussian_derivative<3>(float sigma, int x)
  {
    return ((x * (x * x - 3 * sigma * sigma)) /
             (sqrtf(2 * CUIMG_PI) * std::pow(sigma, 7.f)))
      * expf(-(x * x) / (2.f * sigma * sigma));
  }

  template <>
  inline float gaussian_derivative<4>(float sigma, int x)
  {
    return ((x*x*x*x - 6.f * x*x * sigma * sigma + 3 * std::pow(sigma, 4.f)) /
             (sqrtf(2.f * CUIMG_PI) * std::pow(sigma, 9.f)))
      * expf(-(x * x) / (2 * sigma * sigma));
  }

  inline float gaussian_derivative(int n, float sigma, int x)
  {
    switch(n)
    {
      case 0: return gaussian_derivative<0>(sigma, x);
      case 1: return gaussian_derivative<1>(sigma, x);
      case 2: return gaussian_derivative<2>(sigma, x);
      case 3: return gaussian_derivative<3>(sigma, x);
      case 4: return gaussian_derivative<4>(sigma, x);
      default: return 0;
    }
  }


  template <int N, int dir , typename WW>
  inline void make_gaussian_kernel_1d(float sigma, WW& out)
  {
    int size = sigma * 6 + 1;
    out.resize(size);
    int x = - size / 2;
    for (unsigned i = 0; i < size; i++)
    {
      i_int2 dp;
      bt_getter<dir>::get(dp) = x;
      bt_getter<(dir + 1) % 2>::get(dp) = 0;

      out.dpoints(i) = dp;
      out.weights(i) = gaussian_derivative(N, sigma, x);
      x++;
    }
  }

  template <typename W>
  inline void make_gaussian_kernel_1d(int N, float sigma, std::vector<W>& out)
  {
    int size = int(sigma * 6 + 1);
    out.resize(size);
    int x = - size / 2;
    for (int i = 0; i < size; i++)
    {
      out[i] = gaussian_derivative(N, sigma, x);
      x++;
    }
  }

  template <typename W>
  inline void make_gaussian_kernel_1d(int N, float sigma, std::vector<W>& out, unsigned size)
  {
    out.resize(size);
    int x = - int(size / 2);
    for (int i = 0; i < int(size); i++)
    {
      out[i] = gaussian_derivative(N, sigma, x);
      x++;
    }
  }

  // template <int dir, typename WW>
  // void make_gaussian_kernel_1d(int N, float sigma, WW& out)
  // {
  //   int size = sigma * 6 + 1;
  //   out.resize(size);
  //   int x = - size / 2;
  //   for (unsigned i = 0; i < size; i++)
  //   {
  //     i_int2 dp;
  //     bt_getter<dir>::get(dp) = x;
  //     bt_getter<(dir + 1) % 2>::get(dp) = 0;

  //     out.dpoints(i) = dp;
  //     out.weights(i) = gaussian_derivative<N>(sigma, x);
  //     x++;
  //   }
  // }

  template <int DIR, typename WW>
  inline void make_gaussian_kernel_1d_n(unsigned n, float sigma, WW& out)
  {
    switch (n)
    {
      case 0: return make_gaussian_kernel_1d<0, DIR, WW>(sigma, out);
      case 1: return make_gaussian_kernel_1d<1, DIR, WW>(sigma, out);
      case 2: return make_gaussian_kernel_1d<2, DIR, WW>(sigma, out);
      case 3: return make_gaussian_kernel_1d<3, DIR, WW>(sigma, out);
      case 4: return make_gaussian_kernel_1d<4, DIR, WW>(sigma, out);
    }
  }

  template <typename WW>
  inline void make_gaussian_kernel_1d(unsigned n, unsigned dir, float sigma, WW& out)
  {
    switch (dir)
    {
      case 0: return make_gaussian_kernel_1d_n<0, WW>(n, sigma, out);
      case 1: return make_gaussian_kernel_1d_n<1, WW>(n, sigma, out);
    }
  }

}

#endif
