#ifndef CUIMG_LOAD_IMAGE_H_
# define CUIMG_LOAD_IMAGE_H_


# include <cuimg/improved_builtin.h>
# include <cuimg/cpu/host_image2d.h>

namespace cuimg
{
  void load_image(const std::string& filename, host_image2d<i_uchar3>& out);
}

#endif
