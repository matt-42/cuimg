#ifndef CUIMG_GPU_GET_COMP_H_
# define CUIMG_GPU_GET_COMP_H_

# include <boost/type_traits/remove_reference.hpp>

# include <cuimg/meta.h>
# include <cuimg/dsl/expr.h>
# include <cuimg/dsl/has.h>
# include <cuimg/dsl/eval.h>
# include <cuimg/dsl/traits.h>



namespace cuimg
{

  template <unsigned N, typename E>
  struct get_comp_ : public expr<get_comp_<N, E> >
  {
    typedef int is_expr;

    typedef typename kernel_type<E>::ret KE;
    typedef typename return_type<KE>::ret e_return_type;
    typedef typename boost::remove_reference<typename return_type<E>::ret>::type::vtype local_return_type;

    get_comp_(E& e)
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
      return cuimg::has(p, e_);
    }

    typename kernel_type<E>::ret e_;
  };

  template <unsigned N>
  struct get_comp
  {
    template <typename E>
    static get_comp_<N, E> run(E& e)
    {
      return get_comp_<N, E>(e);
    }
  };

  template <unsigned N, typename E>
  struct return_type<get_comp_<N, E> > { typedef typename boost::remove_reference<typename return_type<E>::ret>::type::vtype ret; };


  template <typename T> struct comp_getters {};

  template <typename TAG, typename E>
  struct get_comp_by_tag_ : public expr<get_comp_by_tag_<TAG, E> >
  {
    typedef int is_expr;
    typedef get_comp_by_tag_<TAG, E> self;
    typedef typename kernel_type<E>::ret KE;
    typedef typename meta::remove_constref<typename return_type<KE>::ret>::ret e_return_type;
    typedef typename return_type<self>::ret local_return_type;

    get_comp_by_tag_(E& e)
      : e_(e)
    {
    }

    __host__ __device__ inline
    local_return_type eval(point2d<int> p) const
    {
      typedef comp_getters<e_return_type> A;
      typedef typename A::template get<TAG> getter;
      return getter::run(cuimg::eval(e_, p));
    }

    __host__ __device__ inline
    bool has(const point2d<int>& p) const
    {
      return cuimg::has(p, e_);
    }

    typename kernel_type<E>::ret e_;
  };

  template <typename TAG, typename E>
  struct return_type<get_comp_by_tag_<TAG, E> >
  {
    typedef typename meta::remove_constref<typename return_type<E>::ret>::ret e_return_type;
    typedef typename comp_getters<e_return_type>::template return_type<TAG> A;
    typedef typename A::ret ret;
  };

  template <typename E>
  get_comp_<0, E> get_x(E& e) { return get_comp_<0, E>(e); }
  template <typename E>
  get_comp_<1, E> get_y(E& e) { return get_comp_<1, E>(e); }
  template <typename E>
  get_comp_<2, E> get_z(E& e) { return get_comp_<2, E>(e); }
  template <typename E>
  get_comp_<3, E> get_w(E& e) { return get_comp_<3, E>(e); }

}

#endif
