#ifndef CUIMG_DRAW_H_
# define CUIMG_DRAW_H_

# include <cuimg/point2d.h>

namespace cuimg
{
  template <typename I>
  __device__ __host__ void plot(I out, point2d<int> a,
                                typename I::value_type color)
  {
    if (c9[0] == 0 && c9[0] == 0)
      for_all_in_static_neighb2d(a, n, c9_h)
      {
        if (out.has(n))
          out(n) = color;
      }
    else
      for_all_in_static_neighb2d(a, n, c9)
      {
        if (out.has(n))
          out(n) = color;
      }
  }

  template <typename I>
  __device__ __host__ void plot_c8(I out, point2d<int> a,
                                   typename I::value_type color)
  {
    if (c8[0] == 0 && c8[0] == 0)
      for_all_in_static_neighb2d(a, n, c8_h)
      {
        if (out.has(n))
          out(n) = color;
      }
    else
      for_all_in_static_neighb2d(a, n, c8)
      {
        if (out.has(n))
          out(n) = color;
      }
  }

  template <typename I>
  __device__ __host__ void draw_line2d(I out, point2d<int> a, point2d<int> b,
                                       typename I::value_type color)
  {
    int x0 = a.col(); int y0 = a.row();
    int x1 = b.col(); int y1 = b.row();

    int steep = abs(y1 - y0) > abs(x1 - x0);

    if (steep)
    {
      swap(x0, y0);
      swap(x1, y1);
    }

    if (x0 > x1)
    {
      swap(x0, x1);
      swap(y0, y1);
    }

    int deltax = x1 - x0;
    int deltay = abs(y1 - y0);
    float error = 0.f;
    float deltaerr = deltay / float(deltax);
    int ystep;
    int y = y0;
    if (y0 < y1) ystep = 1; else ystep = -1;

    for (int x = x0 + 1; x <= x1 - 1; x++)
    {
      if (steep) plot(out, point2d<int>(x, y), color); else plot(out, point2d<int>(y, x), color);
      error = error + deltaerr;
      if (error >= 0.5)
      {
        y = y + ystep;
        error = error - 1.0;
      }
    }
  }

}

#endif
