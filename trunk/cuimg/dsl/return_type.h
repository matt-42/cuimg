#ifndef CUIMG_GPU_RETURN_TYPE_H_
# define CUIMG_GPU_RETURN_TYPE_H_

# include <boost/type_traits/remove_reference.hpp>
# include <cuimg/dsl/expr.h>

namespace cuimg
{


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
    typedef typename return_type_selector<T, is_expr_trait<T>::value>::ret ret;
  };

  template <typename T>
  struct return_type<const T>
  {
    typedef typename return_type<T>::ret ret;
  };
}

#endif
