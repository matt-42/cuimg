#ifndef CUIMG_GPU_GET_COMP_H_
# define CUIMG_GPU_GET_COMP_H_

# include <boost/type_traits/remove_reference.hpp>

# include <cuimg/meta.h>
# include <cuimg/dsl/expr.h>
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

#define ADAPT_STRUCT_BASE_TPL(N, TPL_ARGS, TPL_ARGS_LIST, MEMBERS)      \
  struct comp_getters_tpl_N                                             \
  {                                                                     \
    template <typename TAG> struct get;                                 \
    template <typename TAG> struct return_type;                         \
    MEMBERS                                                             \
   };                                                                   \
  template <TPL_ARGS>                                                   \
  struct comp_getters<N<TPL_ARGS_LIST> > : public comp_getters_tpl_N    \
  {};

#define ADAPT_STRUCT_BASE(N, MEMBERS)           \
  template <>                                   \
  struct comp_getters<N>                        \
  {                                             \
    template <typename TAG> struct get;         \
    template <typename TAG> struct return_type; \
    MEMBERS                                 \
   };


#define ADAPT_STRUCT_MEMBER(S, T, N)                                    \
  template <> struct get<tag::N> { static __host__ __device__ T run(const S& e) { return e.N; } }; \
  template <> struct return_type<tag::N> { typedef T ret; };

#define ADAPT_STRUCT_MAKER(S, ARGS_LIST, VALUES_LIST)     \
  static inline __host__ __device__ S make ARGS_LIST        \
  { return S VALUES_LIST; }

#define ADAPT_STRUCT_1(N, T1, A1)                       \
  ADAPT_STRUCT_BASE(N,                                  \
                    ADAPT_STRUCT_MEMBER(N, T1, A1)); \
    __host__ __device__ N make_##N(T1 A1) \
    { return N(A1); }

#define ADAPT_STRUCT_2(N, T1, A1, T2, A2) \
  ADAPT_STRUCT_BASE(N,                                                  \
                    ADAPT_STRUCT_MEMBER(N, T1, A1)                      \
                    ADAPT_STRUCT_MEMBER(N, T2, A2)                      \
                    )                                                   \
    __host__ __device__ N make_##N(T1 A1, T2 A2) \
    { return N(A1, A2); }

#define ADAPT_STRUCT_3(N, T1, A1, T2, A2, T3, A3) \
  ADAPT_STRUCT_BASE(N,                                                  \
                    ADAPT_STRUCT_MAKER(N, (T1 A1, T2 A2, T3 A3), (A1, A2, A3)) \
                    ADAPT_STRUCT_MEMBER(N, T1, A1)                      \
                    ADAPT_STRUCT_MEMBER(N, T2, A2)                      \
                    ADAPT_STRUCT_MEMBER(N, T3, A3)                      \
                    )                                                   \


/* #define ADAPT_STRUCT_TPL_3(N, T1, A1, T2, A2, T3, A3) \ */
/*   ADAPT_STRUCT_BASE_TPL(N,                                                  \ */
/*                     ADAPT_STRUCT_MAKER(N, (T1 A1, T2 A2, T3 A3), (A1, A2, A3)) \ */
/*                     ADAPT_STRUCT_MEMBER(N, T1, A1)                      \ */
/*                     ADAPT_STRUCT_MEMBER(N, T2, A2)                      \ */
/*                     ADAPT_STRUCT_MEMBER(N, T3, A3)                      \ */
/*                     )                                                   \ */

#define ADAPT_STRUCT_4(N, T1, A1, T2, A2, T3, A3, T4, A4) \
  ADAPT_STRUCT_BASE(N,                                                  \
                    ADAPT_STRUCT_MEMBER(N, T1, A1)                      \
                    ADAPT_STRUCT_MEMBER(N, T2, A2)                      \
                    ADAPT_STRUCT_MEMBER(N, T3, A3)                      \
                    ADAPT_STRUCT_MEMBER(N, T4, A4)                      \
                    )                                                   \
    __host__ __device__ N make_##N(T1 A1, T2 A2, T3 A3, T4 A4) \
    { return N(A1, A2, A3, A4); }

#define ADAPT_STRUCT_5(N, T1, A1, T2, A2, T3, A3, T4, A4, T5, A5) \
  ADAPT_STRUCT_BASE(N,                                                  \
                    ADAPT_STRUCT_MEMBER(N, T1, A1)                      \
                    ADAPT_STRUCT_MEMBER(N, T2, A2)                      \
                    ADAPT_STRUCT_MEMBER(N, T3, A3)                      \
                    ADAPT_STRUCT_MEMBER(N, T4, A4)                      \
                    ADAPT_STRUCT_MEMBER(N, T5, A5)                      \
                    )                                                   \
    __host__ __device__ N make_##N(T1 A1, T2 A2, T3 A3, T4 A4, T5 A5) \
    { return N(A1, A2, A3, A4, A5); }

#define ADAPT_STRUCT_6(N, T1, A1, T2, A2, T3, A3, T4, A4, T5, A5, T6, A6) \
  ADAPT_STRUCT_BASE(N,                                                  \
                    ADAPT_STRUCT_MEMBER(N, T1, A1)                      \
                    ADAPT_STRUCT_MEMBER(N, T2, A2)                      \
                    ADAPT_STRUCT_MEMBER(N, T3, A3)                      \
                    ADAPT_STRUCT_MEMBER(N, T4, A4)                      \
                    ADAPT_STRUCT_MEMBER(N, T5, A5)                      \
                    ADAPT_STRUCT_MEMBER(N, T6, A6)                      \
                    )                                                   \
    __host__ __device__ N make_##N(T1 A1, T2 A2, T3 A3, T4 A4, T5 A5, T6 A6) \
    { return N(A1, A2, A3, A4, A5, A6); }

#define DEFINE_GETTER(TAG) \
  namespace tag { struct TAG {}; } \
  template <typename E>            \
  get_comp_by_tag_<tag::TAG, E> get_##TAG(E& e) { return get_comp_by_tag_<tag::TAG, E>(e); }

DEFINE_GETTER(speed)
DEFINE_GETTER(disp)
DEFINE_GETTER(age)

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
