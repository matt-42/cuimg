#ifndef CUIMG_DRAW_H_
# define CUIMG_DRAW_H_

# include <cuimg/gpu/cuda.h>
# include <cuimg/point2d.h>
# include <cuimg/neighb2d.h>
# include <cuimg/box2d.h>

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

  template <typename I, typename V>
    __host__ __device__ void draw_c8(I& out, const point2d<int>& p, const V& value)
  {
    for_all_in_static_neighb2d(p, n, c8_h) if (out.has(n))
      out(n) = value;
  }

  template <typename I, typename V>
    __host__ __device__ void draw_c8_cuda(I& out, const point2d<int>& p, const V& value)
  {
    for_all_in_static_neighb2d(p, n, c8) if (out.has(n))
      out(n) = value;
  }

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

  template <typename T>
  void draw_abs(T& a, T& b)
  {
    T tmp = a;
    a = b;
    b = tmp;
  }

  template <typename I>
  __device__ __host__ void draw_line2d(I out, point2d<int> a, point2d<int> b,
                                       typename I::value_type color)
  {
    int x0 = a.col(); int y0 = a.row();
    int x1 = b.col(); int y1 = b.row();

    int steep = ::abs(y1 - y0) > ::abs(x1 - x0);

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
    int deltay = ::abs(y1 - y0);
    float error = 0.f;
    float deltaerr = deltay / float(deltax);
    int ystep;
    int y = y0;
    if (y0 < y1) ystep = 1; else ystep = -1;

    for (int x = x0 + 1; x <= x1; x++)
    {
      point2d<int> to_plot;
      if (steep)
	to_plot = point2d<int>(x, y);
      else
	to_plot = point2d<int>(y, x);

      if (out.has(to_plot))
	out(to_plot) = color;

      error = error + deltaerr;
      if (error >= 0.5)
      {
        y = y + ystep;
        error = error - 1.0;
      }
    }
  }

  template <typename I, typename V>
    __host__ __device__ void draw_box2d(I& out, const box2d& bb, const V& value)
  {
    i_int2 diag = bb.p2() - bb.p1();
    i_int2 a(bb.p1());
    i_int2 b(a + i_int2(diag.r(), 0));
    i_int2 c(b + i_int2(0 , diag.c()));
    i_int2 d(a + i_int2(0 , diag.c()));

    draw_line2d(out, a, b, value);
    draw_line2d(out, b, c, value);
    draw_line2d(out, d, c, value);
    draw_line2d(out, d, a, value);
  }

}

#endif
