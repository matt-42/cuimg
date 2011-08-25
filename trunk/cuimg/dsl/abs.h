#ifndef CUIMG_GPU_ABS_H_
# define CUIMG_GPU_ABS_H_

# include <boost/type_traits/remove_reference.hpp>

# include <cmath>
# include <cuimg/meta.h>
# include <cuimg/dsl/expr.h>
# include <cuimg/dsl/eval.h>
# include <cuimg/dsl/traits.h>

# include <cuimg/builtin_math.h>



namespace cuimg
{


  template <typename E>
  struct abs_ : public expr<abs_<E> >
  {
    typedef int is_expr;

    typedef typename kernel_type<E>::ret KE;
    typedef typename return_type<KE>::ret e_return_type;
    typedef e_return_type local_return_type;

    abs_(const E& e)
      : e_(e)
    {
    }

    __host__ __device__ inline
    local_return_type eval(point2d<int> p) const
    {
      return ::abs(cuimg::eval(e_, p));
    }

    __host__ __device__ inline
    bool has(const point2d<int>& p) const
    {
      return cuimg::has(p, e_);
    }

    typename kernel_type<E>::ret e_;
  };

  template <typename E>
  struct return_type<abs_<E> > { typedef float ret; };

  template <typename E>
  abs_<E> abs(const E& e)
  {
    return abs_<E>(e);
  }


}

#endif
