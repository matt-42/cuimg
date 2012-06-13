#ifndef CUIMG_GPU_BINARY_OP_H_
# define CUIMG_GPU_BINARY_OP_H_

# include <cuimg/meta.h>
# include <cuimg/dsl/expr.h>
# include <cuimg/dsl/traits.h>
# include <cuimg/dsl/eval.h>
# include <cuimg/dsl/has.h>

namespace cuimg
{
  template <typename TAG>
  struct evaluator;

  template <typename TAG, typename A1, typename A2, unsigned P1, unsigned P2>
  struct binary_op : public expr<binary_op<typename meta::id<TAG>::ret, A1, A2, P1, P2> >
  {
  public:
    typedef binary_op<TAG, A1, A2, P1, P2> self;
    typedef int is_expr;
    typedef typename return_type<self>::ret return_type;

    binary_op(const A1& a1_, const A2& a2_)
      : a1(a1_),
        a2(a2_)
    {
    }

    __host__ __device__ inline
    return_type
    eval(point2d<int> p) const
    {
      typedef typename cuimg::evaluator<TAG> EV;
      return EV::run(a1, a2, p);
    }

    __host__ __device__ inline
    bool has(const point2d<int>& p) const
    {
      return cuimg::has(p, a1, a2);
    }

    char padding1[P1];
    const typename kernel_type<A1>::ret a1;
    char padding2[P2];
    const typename kernel_type<A2>::ret a2;
  };

  template <typename TAG, typename A1, typename A2, unsigned P1>
  struct binary_op<TAG, A1, A2, P1, 0> : public expr<binary_op<typename meta::id<TAG>::ret, A1, A2, P1, 0> >
  {
  public:
    typedef binary_op<TAG, A1, A2, P1, 0> self;
    typedef int is_expr;
    typedef typename return_type<self>::ret return_type;

    binary_op(const A1& a1_, const A2& a2_)
      : a1(a1_),
        a2(a2_)
    {
    }

    __host__ __device__ inline
    return_type
    eval(point2d<int> p) const
    {
      typedef typename cuimg::evaluator<TAG> EV;
      return EV::run(a1, a2, p);
    }

    __host__ __device__ inline
    bool has(const point2d<int>& p) const
    {
      return cuimg::has(p, a1, a2);
    }

    char padding1[P1];
    const typename kernel_type<A1>::ret a1;
    const typename kernel_type<A2>::ret a2;
  };

  template <typename TAG, typename A1, typename A2, unsigned P2>
  struct binary_op<TAG, A1, A2, 0, P2> : public expr<binary_op<typename meta::id<TAG>::ret, A1, A2, 0, P2> >
  {
  public:
    typedef binary_op<TAG, A1, A2, 0, P2> self;
    typedef int is_expr;
    typedef typename return_type<self>::ret return_type;

    binary_op(const A1& a1_, const A2& a2_)
      : a1(a1_),
        a2(a2_)
    {
    }

    __host__ __device__ inline
    return_type
    eval(point2d<int> p) const
    {
      typedef typename cuimg::evaluator<TAG> EV;
      return EV::run(a1, a2, p);
    }

    __host__ __device__ inline
    bool has(const point2d<int>& p) const
    {
      return cuimg::has(p, a1, a2);
    }

    const typename kernel_type<A1>::ret a1;
    char padding1[P2];
    const typename kernel_type<A2>::ret a2;
  };

  template <typename TAG, typename A1, typename A2>
  struct binary_op<TAG, A1, A2, 0, 0> : public expr<binary_op<typename meta::id<TAG>::ret, A1, A2, 0, 0> >
  {
  public:
    typedef binary_op<TAG, A1, A2, 0, 0> self;
    typedef int is_expr;
    typedef typename return_type<self>::ret return_type;

    binary_op(const A1& a1_, const A2& a2_)
      : a1(a1_),
        a2(a2_)
    {
    }

    __host__ __device__ inline
    return_type
    eval(point2d<int> p) const
    {
      typedef typename cuimg::evaluator<TAG> EV;
      return EV::run(a1, a2, p);
    }

    __host__ __device__ inline
    bool has(const point2d<int>& p) const
    {
      return cuimg::has(p, a1, a2);
    }

    const typename kernel_type<A1>::ret a1;
    const typename kernel_type<A2>::ret a2;
  };

  template <typename TAG, typename A, typename B, unsigned OFFSET = 0>
  struct binary_op_
  {
    typedef typename kernel_type<A>::ret a1;
    typedef typename kernel_type<B>::ret a2;
    enum
    {
//      offset_a1 = __alignof(a1) - ((__alignof(a1) - 1 + OFFSET) % __alignof(a1)) - 1,
//      offset_a2 = __alignof(a2) - ((__alignof(a2) - 1 + (OFFSET + offset_a1 + sizeof(a1))) % __alignof(a2)) - 1
      offset_a1 = 16 - ((16 - 1 + OFFSET) % 16) - 1,
      offset_a2 = 16 - ((16 - 1 + (OFFSET + offset_a1 + sizeof(a1))) % 16) - 1
    };

    typedef binary_op<TAG, A, B, offset_a1, offset_a2> ret;
//    typedef binary_op<TAG, A, B, 0, 0> ret;
  };

}

#endif
