#ifndef CUIMG_TUPLE_EVAL
# define CUIMG_TUPLE_EVAL

# include <cuimg/dsl/tuple.h>

namespace cuimg
{

  template <typename RET, typename A1>
  inline __host__ __device__
  RET tuple_eval(RET (*fun)(A1), tuple<A1> t, point2d<int> p)
  { return fun(eval(t.template get<0>(), p)); }

  template <typename RET, typename A1, typename A2>
  inline __host__ __device__
  RET tuple_eval(RET (*fun)(A1, A2), tuple<A1, A2> t, point2d<int> p)
  { return fun(eval(t.template get<0>(), p), eval(t.template get<1>(), p)); }

  template <typename RET, typename A1,
            typename B1>
  inline __host__ __device__
  RET tuple_eval(RET (*fun)(A1), tuple<B1> t, point2d<int> p)
  { return fun(eval(t.template get<0>(), p)); }

  template <typename RET, typename A1, typename A2,
            typename B1, typename B2>
  inline __host__ __device__
  RET tuple_eval(RET (*fun)(A1, A2), tuple<B1, B2> t, point2d<int> p)
  { return fun(eval(t.template get<0>(), p), eval(t.template get<1>(), p)); }

  template <typename RET, typename A1, typename A2, typename A3,
            typename B1, typename B2, typename B3>
  inline __host__ __device__
  RET tuple_eval(RET (*fun)(A1, A2, A3), tuple<B1, B2, B3> t, point2d<int> p)
  { return fun(eval(t.template get<0>(), p), eval(t.template get<1>(), p), eval(t.template get<2>(), p)); }

  template <typename RET, typename A1, typename A2, typename A3, typename A4,
            typename B1, typename B2, typename B3, typename B4>
  inline __host__ __device__
    RET tuple_eval(RET (*fun)(A1, A2, A3, A4), tuple<B1, B2, B3, B4> t, point2d<int> p)
  { return fun(eval(t.template get<0>(), p), eval(t.template get<1>(), p), eval(t.template get<2>(), p), eval(t.template get<3>(), p)); }


  template <typename RET, typename A1, typename A2, typename A3, typename A4, typename A5,
            typename B1, typename B2, typename B3, typename B4, typename B5>
  inline __host__ __device__
    RET tuple_eval(RET (*fun)(A1, A2, A3, A4, A5), tuple<B1, B2, B3, B4, B5> t, point2d<int> p)
  { return fun(eval(t.template get<0>(), p), eval(t.template get<1>(), p), eval(t.template get<2>(), p), eval(t.template get<3>(), p), eval(t.template get<4>(), p)); }


}

#endif
