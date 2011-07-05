#ifndef CUIMG_GPU_TRAITS_H_
# define CUIMG_GPU_TRAITS_H_

# include <cuimg/gpu/arith/expr.h>
# include <cuimg/improved_builtin.h>

namespace cuimg
{


  template <typename E>
  struct arith_trait
  {
    typedef char one;
    typedef struct { char a[2]; } two;
    typedef struct { char a[3]; } three;

    template <typename T>
    static one sfinae(T* e, typename T::is_expr* v = (typename T::is_expr*)0);

    template <typename T>
    static two sfinae(T* img, typename T::value_type* v = (typename T::value_type*)0);

    static three sfinae(...);

    enum { value = sizeof(sfinae((E*)0)) };
  };


  template <typename A, typename B>
  struct first
  {
    typedef A ret;
  };

  template <typename T, int is_expr>
  struct return_type_selector;

  /* template <typename T> */
  /* struct return_type_selector<T, 1> */
  /* { */
  /*   typedef typename T::return_type ret; */
  /* }; */

  template <typename T>
  struct return_type_selector<T, 2>
  {
    typedef const typename T::value_type& ret;
  };

  template <typename T>
  struct return_type_selector<T, 3>
  {
    typedef T ret;
  };

  template <typename T>
  struct return_type
  {
    typedef typename return_type_selector<T, arith_trait<T>::value>::ret ret;
  };

  template <typename T>
  struct return_type<const T>
  {
    typedef typename return_type<T>::ret ret;
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


  template <typename T>
  struct kernel_type
  {
    typedef typename T ret;
  };

  template <typename I, template <class> class PT>
  struct kernel_type<image2d<I, PT> >
  {
    typedef kernel_image2d<I> ret;
  };

  template <typename I, template <class> class PT>
  struct kernel_type<const image2d<I, PT> >
  {
    typedef kernel_image2d<I> ret;
  };

}

#endif
