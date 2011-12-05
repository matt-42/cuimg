#ifndef CUIMG_GPU_SQRT_H_
# define CUIMG_GPU_SQRT_H_

# include <boost/type_traits/remove_reference.hpp>

# include <cuimg/meta.h>
# include <cuimg/dsl/expr.h>
# include <cuimg/dsl/eval.h>
# include <cuimg/dsl/traits.h>



namespace cuimg
{

  template <typename E>
  struct sqrt_ : public expr<sqrt_<E> >
  {
    typedef int is_expr;

    typedef typename kernel_type<E>::ret KE;
    typedef typename return_type<KE>::ret e_return_type;
    typedef typename boost::remove_reference<typename return_type<E>::ret>::type local_return_type;

    sqrt_(const E& e)
      : e_(e)
    {
    }

    __host__ __device__ inline
    local_return_type eval(point2d<int> p) const
    {
      return ::sqrt(cuimg::eval(e_, p));
    }

    __host__ __device__ inline
    bool has(const point2d<int>& p) const
    {
      return cuimg::has(p, e_);
    }

    typename kernel_type<E>::ret e_;
  };


  template <typename E>
  struct return_type<sqrt_<E> > { typedef typename sqrt_<E>::local_return_type ret; };

  template <typename E>
  sqrt_<E> sqrt(const E& e)
  {
    return sqrt_<E>(e);
  }

}

#endif
