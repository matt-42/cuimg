#ifndef CUIMG_MAP_H_
# define CUIMG_MAP_H_

# include <cuimg/cpu/host_image2d.h>

namespace cuimg
{

  template <typename T, typename U, typename F>
  void map(const host_image2d<T>& in, host_image2d<U>& out, F f)
  {
    assert(in.domain() == out.domain());
    for (unsigned r = 0; r < in.nrows(); r++)
    for (unsigned c = 0; c < in.ncols(); c++)
    {
      out(r, c) = f(in(r, c));
    }
  }

  template <typename T, typename F>
  void map(host_image2d<T>& in, F f)
  {
    map(in, in, f);
  }

}

#endif
