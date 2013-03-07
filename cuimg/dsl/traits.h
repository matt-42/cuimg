#ifndef CUIMG_GPU_TRAITS_H_
# define CUIMG_GPU_TRAITS_H_

# include <cuimg/kernel_type.h>
# include <cuimg/dsl/expr.h>
# include <cuimg/gpu/device_image2d.h>
# include <cuimg/cpu/host_image2d.h>
# include <cuimg/improved_builtin.h>
# include <cuimg/dsl/tuple.h>

namespace cuimg
{


  template <typename E>
  struct is_expr_trait
  {
    typedef char one;
    typedef struct { char a[3]; } three;

    template <typename T>
    static one sfinae(T* e, typename T::is_expr = 0);

    static three sfinae(...);

    enum { value = sizeof(sfinae((E*)0)) };
  };

  template <typename A, typename B, typename C = void, typename D = void>
  struct first
  {
    typedef A ret;
  };

  template <typename T, unsigned PADDING>
  struct padded_member
  {
  private:
    unsigned char align_padding[PADDING];

  public:
    __host__ __device__ padded_member(const T& m_)
      : m(m_)
    {
    }

    __host__ __device__ padded_member<T, PADDING>& operator=(const padded_member<T, PADDING>& o)
    {
      m = o.m;
      return *this;
    }

    T m;
  };

  template <typename T>
  struct padded_member<T, 0>
  {
  public:
    __host__ __device__ padded_member(const T& m_)
      : m(m_)
    {
    }

    T m;
  };

  template <typename T, unsigned OFFSET>
  struct align_member
  {
    typedef padded_member<T, __alignof(T) - ((2*__alignof(T) - 1 + OFFSET) % __alignof(T)) - 1> ret;
  };

  template <typename I>  struct kernel_type<device_image2d<I> >       { typedef kernel_image2d<I> ret; };
  template <typename I>  struct kernel_type<const device_image2d<I> > { typedef kernel_image2d<I> ret; };
  template <typename I>  struct kernel_type<host_image2d<I> >         { typedef kernel_image2d<I> ret; };
  template <typename I>  struct kernel_type<const host_image2d<I> >   { typedef kernel_image2d<I> ret; };

  template <typename A1, typename A2, typename A3,
            typename A4, typename A5, typename A6>
  struct kernel_type<tuple<A1, A2, A3, A4, A5, A6> >
  {
    typedef tuple<typename kernel_type<A1>::ret,
                  typename kernel_type<A2>::ret,
                  typename kernel_type<A3>::ret,
                  typename kernel_type<A4>::ret,
                  typename kernel_type<A5>::ret,
                  typename kernel_type<A6>::ret> ret;
  };

  template <unsigned X>
  struct is_expr_default_void_
  {
    typedef int ret;
  };

  template <>
  struct is_expr_default_void_<3>
  {
    typedef void ret;
  };

  template <typename T>
  struct is_expr_default_void
  {
    typedef typename is_expr_default_void_<is_expr_trait<T>::value>::ret ret;
  };


}

#endif
