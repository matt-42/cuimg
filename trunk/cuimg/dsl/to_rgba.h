#ifndef CUIMG_GPU_TO_RGBA_H_
# define CUIMG_GPU_TO_RGBA_H_

# include <cuimg/dsl/expr.h>
# include <cuimg/dsl/eval.h>
# include <cuimg/meta.h>

namespace cuimg
{

  template <typename T>
  struct to_rgba : public expr<to_rgba<T> >
  {
    typedef int is_expr;

    to_rgba(kernel_image2d<improved_builtin<T, 1> > img)
      : img_(img)
    {
    }

    __host__ __device__ inline
    improved_builtin<T, 4> eval(point2d<int> p) const
    {
      const float x = img_(p).x;
      return improved_builtin<T, 4>(x, x, x, 1.f);
    }

    __host__ __device__ inline
    bool has(const point2d<int>& p) const
    {
      return cuimg::has(p, img_);
    }

    kernel_image2d<improved_builtin<T, 1> > img_;
  };

  template <typename T>
  struct return_type<to_rgba<T> > { typedef improved_builtin<T, 4> ret; };


}

#endif
