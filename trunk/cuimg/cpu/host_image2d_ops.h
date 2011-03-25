#ifndef CUIMG_HOST_IMAGE2D_OPS_H_
# define CUIMG_HOST_IMAGE2D_OPS_H_

# include <cuimg/cpu/host_image2d.h>
# include <cuimg/cpu/map.h>

namespace cuimg
{


  template <typename S>
  struct scalar_add
  {
    scalar_add(const S& s_)
    : s(s_)
    {
    }

    template <typename T>
    T operator()(const T& x)
    {
      return x + s;
    }

  private:
    S s;
  };

  template <typename S>
  struct scalar_mul
  {
    scalar_mul(const S& s_)
    : s(s_)
    {
    }

    template <typename T>
    T operator()(const T& x)
    {
      return x * s;
    }

  private:
    S s;
  };

  template <typename T, typename S>
  void mul(host_image2d<T>& in, const S& s)
  {
    map(in, scalar_mul<S>(s));
  }

  template <typename T, typename S>
  void add(host_image2d<T>& in, const S& s)
  {
    map(in, scalar_add<S>(s));
  }

}

#endif
