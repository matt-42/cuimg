#ifndef CUIMG_GPU_THRESHOLD_H_
# define CUIMG_GPU_THRESHOLD_H_

# include <boost/type_traits/remove_reference.hpp>

# include <cuimg/meta.h>
# include <cuimg/gpu/arith/expr.h>
# include <cuimg/gpu/arith/eval.h>
# include <cuimg/gpu/arith/traits.h>



namespace cuimg
{

  template <typename E, typename S, typename V1, typename V2>
  struct threshold_ : public expr<threshold_<E, S, V1, V2> >
  {
    typedef int is_expr;
    typedef typename return_type<threshold_<E, S, V1, V2> >::ret ret_type;

    threshold_(E e, S s, V1 one, V2 zero)
      : e_(e), s_(s),
        one_(one), zero_(zero)
    {
    }

    __host__ __device__ inline
    ret_type eval(point2d<int> p) const
    {
      if (cuimg::eval(e_, p) < cuimg::eval(s_, p))
        return cuimg::eval(zero_, p);
      else
        return cuimg::eval(one_, p);
     }

    __host__ __device__ inline
    bool has(const point2d<int>& p) const
    {
      return cuimg::has(p, e_, s_, one_, zero_);
    }

    typename kernel_type<E>::ret e_;
    typename kernel_type<S>::ret s_;
    typename kernel_type<V1>::ret one_;
    typename kernel_type<V2>::ret zero_;
  };

  template <typename E, typename S, typename V1, typename V2>
//  struct return_type<threshold_<E, S, V> > { typedef typename boost::remove_reference<V>::type ret; };
  struct return_type<threshold_<E, S, V1, V2> > { typedef typename return_type<typename boost::remove_reference<V1>::type>::ret ret; };

  template <typename E, typename S, typename V1, typename V2>
  threshold_<E, S, V1, V2> threshold(E e, S s, V1 one, V2 zero)
  {
    return threshold_<E, S, V1, V2>(e, s, one, zero);
  }
}

#endif
