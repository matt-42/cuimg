#ifndef CUIMG_TUPLE
# define CUIMG_TUPLE

# include <cuimg/dsl/typelist.h>

namespace cuimg
{

  template <typename A1, typename A2 = null_t, typename A3 = null_t,
            typename A4 = null_t, typename A5 = null_t, typename A6 = null_t>
  struct tuple
  {
  public:
    typedef tuple<A1, A2, A3, A4, A5, A6> self;
    typedef typename make_typelist<A1, A2, A3, A4, A5, A6>::ret list;

    template <typename TU>
    inline __host__ __device__ tuple(TU t)
    {
      *this = t;
    }

    inline __host__ __device__ tuple(A1 a1)
      : head(a1), tail(null_t(), null_t(), null_t(), null_t(), null_t(), null_t()) {}
    inline __host__ __device__ tuple(A1 a1, A2 a2)
      : head(a1), tail(a2, null_t(), null_t(), null_t(), null_t()) {}
    inline __host__ __device__ tuple(A1 a1, A2 a2, A3 a3)
      : head(a1), tail(a2, a3, null_t(), null_t(), null_t()) {}
    inline __host__ __device__ tuple(A1 a1, A2 a2, A3 a3, A4 a4)
      : head(a1), tail(a2, a3, a4, null_t(), null_t()) {}
    inline __host__ __device__ tuple(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5)
      : head(a1), tail(a2, a3, a4, a5, null_t()) {}
    inline __host__ __device__ tuple(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6)
      : head(a1), tail(a2, a3, a4, a5, a6) {}

    template <typename A1, typename A2, typename A3,
              typename A4, typename A5, typename A6>
    inline __host__ __device__
    self& operator=(const tuple<A1, A2, A3, A4, A5, A6>& t)
    {
      head = t.head;
      tail = t.tail;
      return *this;
    }

    template <unsigned I>
    inline __host__ __device__
    typename list_get<list, I>::ret get()
    {
      return tail.template get<I-1>();
    }

    template <>
    inline __host__ __device__
    typename list_get<list, 0>::ret get<0>()
    {
      return head;
    }

    A1 head;
    tuple<A2, A3, A4, A5, A6> tail;
  };

  template <typename A1>
  struct tuple<A1, null_t, null_t, null_t, null_t, null_t>
  {
  public:
    typedef tuple<A1, null_t, null_t, null_t, null_t, null_t> self;
    typedef typename make_typelist<A1>::ret list;

    template <typename TU>
    inline __host__ __device__ tuple(const TU& t)
    {
      *this = t;
    }

    inline __host__ __device__
    tuple(A1 a1, null_t a2 = null_t(), null_t a3 = null_t(), null_t a4 = null_t(),
          null_t a5 = null_t(), null_t a6 = null_t())
      : head(a1)
    {
    }

    template <typename A1>
    inline __host__ __device__
    self& operator=(const tuple<A1, null_t, null_t, null_t, null_t, null_t>& t)
    {
      head = t.head;
      return *this;
    }

    template <unsigned i>
    inline __host__ __device__
    typename A1 get()
    {
      void * a = tuple<A1>(); // Should not be compiled.
      assert(0);
      return head;
    }

    template <>
    inline __host__ __device__
    typename A1 get<0>()
    {
      return head;
    }

    A1 head;
  };

  /*!
  ** Pass tuple members as \p fun parameters.
  ** example:
  **        float fun(float a, int b);
  **
  **        tuple_caller(fun, tuple<float, int>(12.4, 34))
  **              will be translated to fun(12.4, 34);
  **
  ** @param fun a function
  ** @param t a tuple
  **
  ** @return fun(t.get<0>(), t.get<1>(), ...)
  */
  template <typename RET, typename A1>
  inline __host__ __device__
  RET tuple_caller(RET (*fun)(A1), tuple<A1> t)
  { return fun(t.template get<0>()); }

  template <typename RET, typename A1, typename A2>
  inline __host__ __device__
  RET tuple_caller(RET (*fun)(A1, A2), tuple<A1, A2> t)
  { return fun(t.template get<0>(), t.template get<1>()); }

  template <typename RET, typename A1, typename A2, typename A3>
  inline __host__ __device__
  RET tuple_caller(RET (*fun)(A1, A2), tuple<A1, A2, A3> t)
  { return fun(t.template get<0>(), t.template get<1>(), t.template get<2>()); }

  template <typename RET, typename A1, typename A2, typename A3, typename A4>
  inline __host__ __device__
  RET tuple_caller(RET (*fun)(A1, A2), tuple<A1, A2, A3, A4> t)
  { return fun(t.template get<0>(), t.template get<1>(), t.template get<2>(),
               t.template get<3>()); }

  template <typename RET, typename A1, typename A2, typename A3, typename A4, typename A5>
  inline __host__ __device__
  RET tuple_caller(RET (*fun)(A1, A2), tuple<A1, A2, A3, A4, A5> t)
  { return fun(t.template get<0>(), t.template get<1>(), t.template get<2>(),
               t.template get<3>(), t.template get<4>()); }

  template <typename RET, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6>
  inline __host__ __device__
  RET tuple_caller(RET (*fun)(A1, A2), tuple<A1, A2, A3, A4, A5, A6> t)
  { return fun(t.template get<0>(), t.template get<1>(), t.template get<2>(),
               t.template get<3>(), t.template get<4>(), t.template get<5>()); }

}

#endif
