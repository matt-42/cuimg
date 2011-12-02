#ifndef CUIMG_GPU_BINARY_ADD_H_
# define CUIMG_GPU_BINARY_ADD_H_

# include <cuimg/gpu/arith/expr.h>
# include <cuimg/gpu/arith/eval.h>
# include <cuimg/meta.h>

namespace cuimg
{
  template <typename A, typename B, unsigned OFFSET = 0>
  struct binary_add : public expr<binary_add<A, B, OFFSET> >
  {
  public:
    typedef int is_expr;
    typedef typename meta::type_add<typename return_type<A>::ret, typename return_type<B>::ret>::ret return_type;

    binary_add(const A& a, const B& b)
      : a_(a),
        b_(b)
    {
    }

    __device__ inline
    return_type
    eval(point2d<int> p) const
    {
      return cuimg::eval(a_.m, p) + cuimg::eval(b_.m, p);
    }

  private:
    const typename kernel_type<A, OFFSET>::ret a_;
    const typename kernel_type<B, OFFSET +
         sizeof(typename kernel_type<A, OFFSET>::ret)>::ret b_;
  };

  template <typename A, unsigned OFFSET>
  struct binary_add<A, i_float4, OFFSET> : public expr<binary_add<A, i_float4, OFFSET> >
  {
  public:
    typedef int is_expr;
    typedef typename meta::type_add<typename return_type<A>::ret, i_float4>::ret return_type;

    binary_add(const A& a, const i_float4& b)
      : a_(a),
        b_(b)
    {
    }

    __device__ inline
    return_type
    eval(point2d<int> p) const
    {
      return b_.m;//i_float4(b_.x, b_.y, b_.z, b_.w);
      //      return cuimg::eval(a_.m, p) + cuimg::eval(b_.m, p);
    }

  private:
    const typename kernel_type<A, OFFSET>::ret a_;
//    const typename kernel_type<i_float4, OFFSET +
//              sizeof(typename kernel_type<A, OFFSET>::ret)>::ret b_;
//    const typename align_member<i_float4, 20>::ret b_;
    const padded_member<i_float4, 12> b_;
  //  unsigned char pad[12];
  //  const i_float4 b_;
  };

  template <typename A, typename B,
  struct binary_add_
  {
    typedef binary_add_<>
  };

  template <typename A, template <class> class AP, typename B, template <class> class BP>
  inline
  binary_add<image2d<A, AP>, image2d<B, BP> >
  operator+(const image2d<A, AP>& a, const image2d<B, BP>& b)
  {
    return binary_add<image2d<A, AP>, image2d<B, BP> >(a, b);
  }

  template <typename A, template <class> class AP, typename S>
  inline
  binary_add<image2d<A, AP>, S>
  operator+(const image2d<A, AP>& a, const S s)
  {
    return binary_add<image2d<A, AP>, S>(a, s);
  }

  template <typename E, typename S>
  inline
  typename first<binary_add<E, S>, typename E::is_expr>::ret
  operator+(E& a, const S s)
  {
    return binary_add<E, S>(a, s);
  }

}

#endif
