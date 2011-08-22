#ifndef CUIMG_COLOR_CONVERSION_H_
# define CUIMG_COLOR_CONVERSION_H_

# include <cuimg/gpu/kernel_image2d.h>

namespace cuimg
{

  template <typename I>
  struct rgb_to_hsv : public expr<rgb_to_hsv<I> >
  {
    typedef int is_expr;
    typedef typename I::value_type value_type;

    rgb_to_hsv(const Image2d<I>& img)
      : img_(exact(img))
    {
    }

    __host__ __device__ inline
    value_type eval(point2d<int> p) const
    {
      value_type in = img_(p);
      value_type res;
      typename value_type::vtype cmax, cmin;

      cmin = min(in.x, in.y);
      cmin = min(cmin, in.z);

      if (in.x > in.y && in.x > in.z) // max == R
      {
        cmax = in.x;
        res.x = (int(60.f * (in.y - in.z) / (cmax-cmin) + 360) % 360) / 360.f;
      }
      else if (in.y > in.x && in.y > in.z) // max == G
      {
        cmax = in.y;
        res.x = (int(60.f * (in.z - in.x) / (cmax-cmin) + 120) % 360) / 360.f;
      }
      else if (in.z > in.x && in.z > in.y) // max == B
      {
        cmax = in.z;
        res.x = (int(60.f * (in.x - in.y) / (cmax-cmin) + 240) % 360) / 360.f;
      }
      else
      {
        cmax = in.x;
        res.x = 0.f;
      }

      if (cmax != 0)
        res.y = 1 - cmin/cmax;
      else
        res.y = 0;

      res.z = cmax;
      return res;
    }

    __host__ __device__ inline
    bool has(const point2d<int>& p) const
    {
      return has(p, img_);
    }

    const typename kernel_type<I>::ret& img_;
  };

  template <typename I>
  struct return_type<rgb_to_hsv<I> > { typedef typename I::value_type ret; };

}


#endif
