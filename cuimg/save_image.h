#ifndef CUIMG_SAVE_IMAGE_H_
# define CUIMG_SAVE_IMAGE_H_


# include <cuimg/improved_builtin.h>
# include <cuimg/cpu/host_image2d.h>

namespace cuimg
{
  void save_image(const std::string& filename, const host_image2d<i_uchar3>& out);
}

#endif
