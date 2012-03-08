#ifndef CUIMG_IMAGE2D_TARGET_H_
# define CUIMG_IMAGE2D_TARGET_H_

# include <cuimg/target.h>

namespace cuimg
{

  template <unsigned T>
  struct make_image2d_target;

  template <>
  struct make_image2d_target<CPU>
  {
    template <typename V>
    struct run
    {
      typedef host_image2d<V> ret;
    };
  };

  template <>
  struct make_image2d_target<GPU>
  {
    template <typename V>
    struct run
    {
      typedef device_image2d<V> ret;
    };
  };

#define image2d_target(T, V) typename make_image2d_target<T>::template run<V>::ret

}

#endif // !CUIMG_TARGET_H_
