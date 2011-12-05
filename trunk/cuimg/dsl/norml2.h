#ifndef CUIMG_GPU_NORML2_H_
# define CUIMG_GPU_NORML2_H_

# include <boost/type_traits/remove_reference.hpp>

# include <cuimg/meta.h>
# include <cuimg/dsl/expr.h>
# include <cuimg/dsl/eval.h>
# include <cuimg/dsl/traits.h>

# include <cuimg/builtin_math.h>



namespace cuimg
{


  template <typename E>
  struct norml2_ : public expr<norml2_<E> >
  {
    typedef int is_expr;

    typedef typename kernel_type<E>::ret KE;
    typedef typename return_type<KE>::ret e_return_type;
    typedef float local_return_type;

    norml2_(const E& e)
      : e_(e)
    {
    }

    __host__ __device__ inline
    local_return_type eval(point2d<int> p) const
    {
      return cuimg::norml2(cuimg::eval(e_, p));
    }

    __host__ __device__ inline
    bool has(const point2d<int>& p) const
    {
      return cuimg::has(p, e_);
    }

    typename kernel_type<E>::ret e_;
  };

  template <typename E>
  struct return_type<norml2_<E> > { typedef float ret; };

  template <typename E>
  norml2_<E> norml2(const E& e)
  {
    return norml2_<E>(e);
  }


}

#endif
