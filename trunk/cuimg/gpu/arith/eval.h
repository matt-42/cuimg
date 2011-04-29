#ifndef CUIMG_GPU_EVAL_H_
# define CUIMG_GPU_EVAL_H_

# include <cuimg/gpu/arith/expr.h>
# include <cuimg/gpu/arith/traits.h>
# include <cuimg/meta.h>

namespace cuimg
{

  template <typename X, int T>
  struct eval_selector
  {
    static inline __device__
    typename return_type<X>::ret
    run(X& x, point2d<int>)
    {
      return x;
    }
  };

  template <typename X>
  struct eval_selector<X, 1>
  {
    static inline __device__
    typename return_type<X>::ret
    run(X& x, point2d<int> p)
    {
      return x.eval(p);
    }
  };

  template <typename X>
  struct eval_selector<X, 2>
  {
    static inline __device__
    typename return_type<X>::ret
    run(const X& x, point2d<int> p)
    {
      return x(p);
    }
  };

  template <typename X>
  inline __device__
  typename return_type<X>::ret
  eval(X& x, point2d<int> p)
  {
    return eval_selector<X, arith_trait<X>::value>::run(x, p);
  }

}

#endif
