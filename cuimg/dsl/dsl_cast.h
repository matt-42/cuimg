#ifndef CUIMG_GPU_DSL_CAST_H_
# define CUIMG_GPU_DSL_CAST_H_

# include <cuimg/dsl/expr.h>
# include <cuimg/dsl/eval.h>
# include <cuimg/dsl/has.h>
# include <cuimg/meta.h>

namespace cuimg
{

  template <typename T, typename E>
  struct dsl_cast_ : public expr<dsl_cast_<T, E> >
  {
    typedef int is_expr;

    dsl_cast_(const E& e)
      : e_(e)
    {
    }

    __host__ __device__ inline
    T eval(point2d<int> p) const
    {
      return T(cuimg::eval(e_, p));
    }

    __host__ __device__ inline
    bool has(const point2d<int>& p) const
    {
      return cuimg::has(p, e_);
    }

    typename kernel_type<E>::ret e_;
  };

  template <typename T, typename E>
  struct return_type<dsl_cast_<T, E> > { typedef T ret; };

  template <typename T>
  struct dsl_cast
  {
    template <typename E>
    static dsl_cast_<T, E> run(const E& e)
    {
      return dsl_cast_<T, E>(e);
    }
  };

}

#endif
