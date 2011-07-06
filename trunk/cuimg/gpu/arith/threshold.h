#ifndef CUIMG_GPU_THRESHOLD_H_
# define CUIMG_GPU_THRESHOLD_H_

# include <boost/type_traits/remove_reference.hpp>

# include <cuimg/meta.h>
# include <cuimg/gpu/arith/expr.h>
# include <cuimg/gpu/arith/eval.h>
# include <cuimg/gpu/arith/traits.h>



namespace cuimg
{

  template <typename E, typename S, typename V>
  struct threshold_ : public expr<threshold_<E, S, V> >
  {
    typedef int is_expr;

    threshold_(E& e, S s, V one, V zero)
      : e_(e), s_(s),
        one_(one), zero_(zero)
    {
    }

    __host__ __device__ inline
    V eval(point2d<int> p) const
    {
      if (cuimg::eval(e_, p) < E::value_type(s_))
        return zero_;
      else
        return one_;
     }

    typename kernel_type<E>::ret e_;
    S s_;
    V one_, zero_;
  };

  template <typename E, typename S, typename V>
  struct return_type<threshold_<E, S, V> > { typedef typename boost::remove_reference<V>::type ret; };

  template <typename E, typename S, typename V>
  threshold_<E, S, V> threshold(E& e, S s, V one, V zero)
  {
    return threshold_<E, S, V>(e, s, one, zero);
  }
}

#endif
