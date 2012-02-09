#ifndef CUIMG_GPU_SUPER_NUMERIC_TYPE_H_
# define CUIMG_GPU_SUPER_NUMERIC_TYPE_H_

# include <cuimg/improved_builtin.h>

namespace cuimg
{

  template <typename T>
  struct super_numeric_type
  {
    typedef float ret;
  };

  template <typename T, unsigned N>
  struct super_numeric_type<improved_builtin<T, N> >
  {
    typedef improved_builtin<typename super_numeric_type<T>::ret, N> ret;
  };

}

#endif
