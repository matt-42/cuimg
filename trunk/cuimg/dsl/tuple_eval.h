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

  template <typename RET, typename A1, typename A2, typename A3,
            typename B1, typename B2, typename B3>
  inline __host__ __device__
  RET tuple_eval(RET (*fun)(A1, A2, A3), tuple<B1, B2, B3> t, point2d<int> p)
  { return fun(eval(t.template get<0>(), p), eval(t.template get<1>(), p), eval(t.template get<2>(), p)); }

  template <typename RET, typename A1, typename A2, typename A3, typename A4>
  inline __host__ __device__
    RET tuple_eval(RET (*fun)(A1, A2, A3, A4), tuple<A1, A2, A3, A4> t, point2d<int> p)
  { return fun(eval(t.template get<0>(), p), eval(t.template get<1>(), p), eval(t.template get<2>(), p),
               eval(t.template get<3>(), p)); }

  template <typename RET, typename A1, typename A2, typename A3, typename A4, typename A5>
  inline __host__ __device__
  RET tuple_eval(RET (*fun)(A1, A2, A3, A4, A5), tuple<A1, A2, A3, A4, A5> t, point2d<int> p)
  { return fun(eval(t.template get<0>(), p), eval(t.template get<1>(), p), eval(t.template get<2>(), p),
               eval(t.template get<3>(), p), eval(t.template get<4>(), p)); }

  template <typename RET, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6>
  inline __host__ __device__
  RET tuple_eval(RET (*fun)(A1, A2, A3, A4, A5, A6), tuple<A1, A2, A3, A4, A5, A6> t, point2d<int> p)
  { return fun(eval(t.template get<0>(), p), eval(t.template get<1>(), p), eval(t.template get<2>(), p),
               eval(t.template get<3>(), p), eval(t.template get<4>(), p), eval(t.template get<5>(), p)); }

}

#endif
