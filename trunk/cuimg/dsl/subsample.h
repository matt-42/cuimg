#ifndef CUIMG_GPU_SUBSAMPLE_H_
# define CUIMG_GPU_SUBSAMPLE_H_

# include <boost/type_traits/remove_reference.hpp>

# include <cuimg/meta.h>
# include <cuimg/dsl/expr.h>
# include <cuimg/dsl/eval.h>
# include <cuimg/dsl/traits.h>



namespace cuimg
{

  template <typename E>
  struct subsample_ : public expr<subsample_<E> >
  {
    typedef int is_expr;
    typedef typename return_type<subsample_<E> >::ret ret_type;

    subsample_(const E& e)
      : e_(e)
    {
    }

    __host__ __device__ inline
    ret_type eval(point2d<int> p) const
    {
      int cpt = 1;
      p = point2d<int>(p.row() * 2, p.col() * 2);

      typedef typename return_type<E>::ret ret_type;
      typedef typename cuimg::meta::type_add<ret_type, ret_type>::ret add_type;

      add_type res = cuimg::eval(e_, p);

      point2d<int> n(p.row() + 1, p.col());
      if (cuimg::has(n, e_))
      {
        res = res + cuimg::eval(e_, n);
        cpt++;
      }

      n = point2d<int>(p.row() + 1, p.col() + 1);
      if (cuimg::has(n, e_))
      {
        res = res + cuimg::eval(e_, n);
        cpt++;
      }

      n = point2d<int>(p.row(), p.col() + 1);
      if (cuimg::has(n, e_))
      {
        res = res + cuimg::eval(e_, n);
        cpt++;
      }

      return res / cpt;
    }

    __host__ __device__ inline
    bool has(const point2d<int>& p) const
    {
      return cuimg::has(p, e_);
    }

    typename kernel_type<E>::ret e_;
  };

  template <typename E>
  struct return_type<subsample_<E> > { typedef typename return_type<E>::ret ret; };

  template <typename E>
  subsample_<E> subsample(const E& e)
  {
    return subsample_<E>(e);
  }
}

#endif
