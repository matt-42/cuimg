#ifndef CUIMG_GPU_RETURN_TYPE_H_
# define CUIMG_GPU_RETURN_TYPE_H_

# include <boost/type_traits/remove_reference.hpp>
# include <cuimg/gpu/arith/expr.h>

namespace cuimg
{


  template <typename E>
  struct return_type_trait
  {
    typedef char one;
    typedef struct { char a[2]; } two;
    typedef struct { char a[3]; } three;

    template <typename T>
    static one sfinae(expr<T>* e);

    template <typename T>
    static two sfinae(T* img, typename T::value_type* v = (typename T::value_type*)0);

    static three sfinae(...);

    enum { value = sizeof(sfinae((E*)0)) };
  };

  template <typename T, int is_expr>
  struct return_type_selector;

  template <typename T>
  struct return_type_selector<T, 2>
  {
    typedef typename T::value_type ret;
  };

  template <typename T>
  struct return_type_selector<T, 3>
  {
    typedef typename boost::remove_reference<T>::type ret;
  };

  template <typename T>
  struct return_type
  {
    typedef typename return_type_selector<T, return_type_trait<T>::value>::ret ret;
  };

}

#endif
