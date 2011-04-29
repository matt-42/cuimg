#ifndef CUIMG_GPU_BINARY_MUL_H_
# define CUIMG_GPU_BINARY_MUL_H_

# include <cuimg/gpu/arith/expr.h>
# include <cuimg/gpu/arith/eval.h>
# include <cuimg/meta.h>

namespace cuimg
{
  template <typename A, typename B>
  struct binary_mul : public expr<binary_mul<A, B> >
  {
  public:
    typedef int is_expr;
    typedef typename meta::type_mult<typename return_type<A>::ret, typename return_type<B>::ret>::ret return_type;

    binary_mul(const A& a, const B& b)
      : a_(a),
        b_(b)
    {
    }

    __device__ inline
    return_type
    eval(point2d<int> p) const
    {
      return cuimg::eval(a_, p) * cuimg::eval(b_, p);
    }

  private:
    const typename kernel_type<A>::ret a_;
    const typename kernel_type<B>::ret b_;
  };

  template <typename A, template <class> class AP, typename S>
  inline
  binary_mul<image2d<A, AP>, S>
  operator*(const image2d<A, AP>& a, const S s)
  {
    return binary_mul<image2d<A, AP>, S>(a, s);
  }

  template <typename E, typename S>
  inline
  typename first<binary_mul<E, S>, typename E::is_expr>::ret
  operator*(E& a, const S& s)
  {
    return binary_mul<E, S>(a, s);
  }

}

#endif
