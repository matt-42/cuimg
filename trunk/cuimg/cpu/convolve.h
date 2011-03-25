#ifndef CUIMG_CONVOLVE_H
# define CUIMG_CONVOLVE_H

# include <cuimg/cpu/host_image2d.h>
# include <cuimg/improved_builtin.h>

namespace cuimg
{
  template <typename V, typename W>
  void convolve_rows(host_image2d<V>& in, host_image2d<V>& out, W& weights)
  {
    int hs = weights.size() / 2;
    for (int r = 0; r < int(in.nrows()); r++)
    for (int c = 0; c < int(in.ncols()); c++)
    {
      V res = zero();
      for (int i = c - hs; i < c + hs; i++)
      {
        if (i >= 0 && i < int(in.ncols()))
        {
          res += in(r, i) * weights[i - (c - hs)];
        }
      }
      out(r, c) = res;
    }
  }

  template <typename V, typename W>
  void convolve_cols(host_image2d<V>& in, host_image2d<V>& out, W& weights)
  {
    int hs = weights.size() / 2;
    for (int r = 0; r < int(in.nrows()); r++)
    for (int c = 0; c < int(in.ncols()); c++)
    {
      V res = zero();
      for (int i = r - hs; i < r + hs; i++)
      {
        if (i >= 0 && i < int(in.nrows()))
        {
          res += in(i, c) * weights[i - (r - hs)];
        }
      }
      out(r, c) = res;
    }
  }

}

#endif
