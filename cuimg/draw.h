#ifndef CUIMG_DRAW_H_
# define CUIMG_DRAW_H_

# include <cuimg/gpu/cuda.h>
# include <cuimg/point2d.h>
# include <cuimg/neighb2d.h>

namespace cuimg
{
  /* template <typename I> */
  /* __device__ __host__ void plot(I out, point2d<int> a, */
  /*                               typename I::value_type color) */
  /* { */
  /*   if (c9[0] == 0 && c9[0] == 0) */
  /*     for_all_in_static_neighb2d(a, n, c9_h) */
  /*     { */
  /*       if (out.has(n)) */
  /*         out(n) = color; */
  /*     } */
  /*   else */
  /*     for_all_in_static_neighb2d(a, n, c9) */
  /*     { */
  /*       if (out.has(n)) */
  /*         out(n) = color; */
  /*     } */
  /* } */

  /* template <typename I> */
  /* __device__ __host__ void plot_c8(I out, point2d<int> a, */
  /*                                  typename I::value_type color) */
  /* { */
  /*   if (c8[0] == 0 && c8[0] == 0) */
  /*     for_all_in_static_neighb2d(a, n, c8_h) */
  /*     { */
  /*       if (out.has(n)) */
  /*         out(n) = color; */
  /*     } */
  /*   else */
  /*     for_all_in_static_neighb2d(a, n, c8) */
  /*     { */
  /*       if (out.has(n)) */
  /*         out(n) = color; */
  /*     } */
  /* } */

  template <typename I>
    __device__ __host__ void fill_rect(I out, point2d<int> a, unsigned nr, unsigned nc,
				       typename I::value_type color)
  {
    for (unsigned r = 0; r < nr; r++)
      for (unsigned c = 0; c < nc; c++)
      {
	point2d<int> n(a.row() + r, a.col() + c);
	if (out.has(n)) out(n) = color;
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
      std::swap(x0, y0);
      std::swap(x1, y1);
    }

    if (x0 > x1)
    {
      std::swap(x0, x1);
      std::swap(y0, y1);
    }

    int deltax = x1 - x0;
    int deltay = abs(y1 - y0);
    float error = 0.f;
    float deltaerr = deltay / float(deltax);
    int ystep;
    int y = y0;
    if (y0 < y1) ystep = 1; else ystep = -1;

    for (int x = x0 + 1; x <= x1; x++)
    {
      if (steep) out(point2d<int>(x, y)) = color; else out(point2d<int>(y, x)) = color;
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
