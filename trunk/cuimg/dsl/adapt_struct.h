#ifndef CUIMG_ADAPT_STRUCT_H_
# define CUIMG_ADAPT_STRUCT_H_

namespace cuimg
{


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
                    ADAPT_STRUCT_MAKER(N, (T1 A1, T2 A2, T3 A3, T4 A4), (A1, A2, A3, A4)) \
                    ADAPT_STRUCT_MEMBER(N, T1, A1)                      \
                    ADAPT_STRUCT_MEMBER(N, T2, A2)                      \
                    ADAPT_STRUCT_MEMBER(N, T3, A3)                      \
                    ADAPT_STRUCT_MEMBER(N, T4, A4)                      \
                    )                                                   \

#define ADAPT_STRUCT_5(N, T1, A1, T2, A2, T3, A3, T4, A4, T5, A5) \
  ADAPT_STRUCT_BASE(N,                                                  \
                    ADAPT_STRUCT_MAKER(N, (T1 A1, T2 A2, T3 A3, T4 A4, T5 A5), (A1, A2, A3, A4, A5)) \
                    ADAPT_STRUCT_MEMBER(N, T1, A1)                      \
                    ADAPT_STRUCT_MEMBER(N, T2, A2)                      \
                    ADAPT_STRUCT_MEMBER(N, T3, A3)                      \
                    ADAPT_STRUCT_MEMBER(N, T4, A4)                      \
                    ADAPT_STRUCT_MEMBER(N, T5, A5)                      \
                    )

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
DEFINE_GETTER(accel)
DEFINE_GETTER(disp)
DEFINE_GETTER(orig)
DEFINE_GETTER(age)
DEFINE_GETTER(to_draw)

}

#endif
