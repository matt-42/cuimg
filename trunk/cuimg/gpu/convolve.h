#ifndef CUIMG_CONVOLVE_H_
# define CUIMG_CONVOLVE_H_

# include <cuimg/gpu/image2d.h>

namespace cuimg
{
  template <typename U, typename V, typename WW>
  void convolve(const image2d<U>& in, image2d<V>& out, const WW& weighted_window, dim3 dimblock = dim3(16, 16, 1));

  template <typename U, typename V, typename WW>
  void convolveRows(const image2d<U>& in, image2d<V>& out, const WW& weighted_window);

  template <typename U, typename V, typename WW>
  void convolveCols(const image2d<U>& in, image2d<V>& out, const WW& weighted_window);

}

# include <cuimg/gpu/convolve.hpp>

#endif
