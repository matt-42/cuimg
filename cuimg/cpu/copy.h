#ifndef CUIMG_COPY_H_
# define CUIMG_COPY_H_

# include <cuimg/cpu/host_image2d.h>
# include <cuimg/cpu/host_image3d.h>

namespace cuimg
{

  template <typename T, typename U>
  void copy(const host_image2d<T>& in, host_image2d<T>& out)
  {
    assert(in.domain() == out.domain());
    for (unsigned r = 0; r < in.nrows(); r++)
    for (unsigned c = 0; c < in.ncols(); c++)
    {
      out(r, c) = in(r, c);
    }
  }

  template <typename T>
  void copy(const host_image2d<T>& in, host_image2d<T>& out)
  {
    assert(in.domain() == out.domain());
    memcpy(out.data(), in.data(), in.ncols() * in.nrows() * sizeof(T));
  }

  template <typename T>
  void copy(const host_image3d<T>& in, host_image3d<T>& out)
  {
    assert(in.domain() == out.domain());
    memcpy(out.data(), in.data(), in.nslices() * in.ncols() * in.nrows() * sizeof(T));
  }

}

#endif
