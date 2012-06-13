#ifndef CUIMG_COLOR_CONVERSION_H_
# define CUIMG_COLOR_CONVERSION_H_

# include <cuimg/gpu/device_image2d.h>

namespace cuimg
{
  template <typename U, typename V, typename WW>
  void rgb_to_hsv(const device_image2d<U>& in, device_image2d<V>& out);

}


#endif
