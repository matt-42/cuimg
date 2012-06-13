#ifndef CUIMG_TUPLE
# define CUIMG_TUPLE

# include <cuimg/dsl/typelist.h>

namespace cuimg
{

  template <typename T, unsigned I>
  struct tuple_getter;

  template <typename A1, typename A2 = null_t, typename A3 = null_t,
            typename A4 = null_t, typename A5 = null_t, typename A6 = null_t>
  struct tuple
  {
  public:
    typedef tuple<A1, A2, A3, A4, A5, A6> self;
    typedef tuple<A2, A3, A4, A5, A6> tail_type;
    typedef A1 head_type;
    typedef typename make_typelist<A1, A2, A3, A4, A5, A6>::ret list;

    template <typename TU>
    inline __host__ __device__ tuple(TU t)
    {
      *this = t;
    }

    inline __host__ __device__ tuple(A1 a1)
      : head_(a1), tail_(null_t(), null_t(), null_t(), null_t(), null_t(), null_t()) {}
    inline __host__ __device__ tuple(A1 a1, A2 a2)
      : head_(a1), tail_(a2, null_t(), null_t(), null_t(), null_t()) {}
    inline __host__ __device__ tuple(A1 a1, A2 a2, A3 a3)
      : head_(a1), tail_(a2, a3, null_t(), null_t(), null_t()) {}
    inline __host__ __device__ tuple(A1 a1, A2 a2, A3 a3, A4 a4)
      : head_(a1), tail_(a2, a3, a4, null_t(), null_t()) {}
    inline __host__ __device__ tuple(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5)
      : head_(a1), tail_(a2, a3, a4, a5, null_t()) {}
    inline __host__ __device__ tuple(A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6)
      : head_(a1), tail_(a2, a3, a4, a5, a6) {}

    template <typename B1, typename B2, typename B3,
              typename B4, typename B5, typename B6>
    inline __host__ __device__
    self& operator=(const tuple<B1, B2, B3, B4, B5, B6>& t)
    {
      head_ = t.head_;
      tail_ = t.tail_;
      return *this;
    }

    template <unsigned I>
    inline __host__ __device__
    typename list_get<list, I>::ret get()
    {
      return tuple_getter<self, I>::run(*this);
    }

    const tail_type& tail() const { return tail_; }
    tail_type& tail() { return tail_; }

    const head_type& head() const { return head_; }
    head_type& head() { return head_; }

    A1 head_;
    tail_type tail_;
  };


  template <typename T, unsigned I>
  struct tuple_getter
  {
    typedef typename list_get<typename T::list, I>::ret return_type;

    static inline __host__ __device__
    const return_type& run(const T& t)
    {
      return tuple_getter<typename T::tail_type, I-1>(t.tail());
    }

    static inline __host__ __device__
    return_type& run(T& t)
    {
      return tuple_getter<typename T::tail_type, I-1>(t.tail());
    }

  };


  template <typename T>
  struct tuple_getter<T, 0>
  {
    typedef typename list_get<typename T::list, 0>::ret return_type;

    static inline __host__ __device__
    const return_type& run(const T& t)
    {
      return t.head();
    }

    static inline __host__ __device__
    return_type& run(T& t)
    {
      return t.head();
    }

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

    template <typename P1>
    inline __host__ __device__
    self& operator=(const tuple<P1, null_t, null_t, null_t, null_t, null_t>& t)
    {
      head = t.head;
      return *this;
    }

    template <unsigned i>
    inline __host__ __device__
    const A1& get() const
    {
      return head;
    }

    template <unsigned i>
    inline __host__ __device__
    A1& get()
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
