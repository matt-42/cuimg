#ifndef CUIMG_GPU_WRAPPER_H_
# define CUIMG_GPU_WRAPPER_H_

# include <boost/type_traits/remove_reference.hpp>

# include <cuimg/meta.h>
# include <cuimg/gpu/arith/expr.h>
# include <cuimg/gpu/arith/eval.h>
# include <cuimg/gpu/arith/traits.h>



namespace cuimg
{

  template <typename F>
  struct wrapper : public expr<wrapper_<N, E> >
  {
    typedef int is_expr;

    typedef typename kernel_type<E>::ret KE;
    typedef typename return_type<KE>::ret e_return_type;
    typedef typename boost::remove_reference<typename return_type<E>::ret>::type::vtype local_return_type;

    wrapper_(E& e)
      : e_(e)
    {
    }

    __host__ __device__ inline
    local_return_type eval(point2d<int> p) const
    {
      return bt_getter<N>::get(cuimg::eval(e_, p));
    }

    __host__ __device__ inline
    bool has(const point2d<int>& p) const
    {
      return has(p, e_);
    }

    typename kernel_type<E>::ret e_;
  };

  template <unsigned N>
  struct wrapper
  {
    template <typename E>
    static wrapper_<N, E> run(E& e)
    {
      return wrapper_<N, E>(e);
    }
  };

  template <unsigned N, typename E>
  struct return_type<wrapper_<N, E> > { typedef typename boost::remove_reference<typename return_type<E>::ret>::type::vtype ret; };

  template <typename E>
  wrapper_<0, E> get_x(E& e) { return wrapper_<0, E>(e); }
  template <typename E>
  wrapper_<1, E> get_y(E& e) { return wrapper_<1, E>(e); }
  template <typename E>
  wrapper_<2, E> get_z(E& e) { return wrapper_<2, E>(e); }
  template <typename E>
  wrapper_<3, E> get_w(E& e) { return wrapper_<3, E>(e); }

}

#endif
