#ifndef CUIMG_GPU_BINARY_ADD_H_
# define CUIMG_GPU_BINARY_ADD_H_

# include <cuimg/gpu/arith/expr.h>
# include <cuimg/gpu/arith/eval.h>
# include <cuimg/meta.h>

namespace cuimg
{
  template <typename A, typename B>
  struct binary_add : public expr<binary_add<A, B> >
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
      return cuimg::eval(a_, p) + cuimg::eval(b_, p);
    }

  private:
    const typename kernel_type<A>::ret a_;
    const typename kernel_type<B>::ret b_;
  };

  template <typename A>
  struct binary_add<A, i_float4> : public expr<binary_add<A, i_float4> >
  {
  public:
    typedef int is_expr;
    typedef typename meta::type_add<typename return_type<A>::ret, i_float4>::ret return_type;

    binary_add(const A& a, const i_float4& b)
      : //a_(a),
        b_(b)
    {
    }

    __device__ inline
    return_type
    eval(point2d<int> p) const
    {
      return b_;
//      return cuimg::eval(a_, p) + b_;
    }

  private:
    //const typename kernel_type<A>::ret a_;
    const float4 b_;
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
