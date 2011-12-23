#ifndef CUIMG_CPU_FILL_H_
# define CUIMG_CPU_FILL_H_

# include <cuimg/improved_builtin.h>
# include <cuimg/cpu/host_image2d.h>
# include <cuimg/error.h>
# include <cuimg/util.h>

namespace cuimg
{

  template<typename U>
  void fill(host_image2d<U>& out,
            const U& v)
  {
    for (unsigned r = 0; r < out.nrows(); r++)
    for (unsigned c = 0; c < out.ncols(); c++)
      out(r, c) = v;
  }

}

#endif
