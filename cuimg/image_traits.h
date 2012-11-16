#ifndef CUIMG_IMAGE_TRAITS_H_
# define CUIMG_IMAGE_TRAITS_H_

# include <cuimg/gpu/cuda.h>

namespace cuimg
{

  template <typename I, typename NV>
  struct change_value_type
  {
  };

  template <typename V, typename NV>
  struct change_value_type<host_image2d<V>, NV>
  {
    typedef host_image2d<NV> ret;
  };

  template <typename V, typename NV>
  struct change_value_type<device_image2d<V>, NV>
  {
    typedef device_image2d<NV> ret;
  };

}

#endif
