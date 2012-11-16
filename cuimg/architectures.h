#ifndef CUIMG_ARCHITECTURES_H_
# define CUIMG_ARCHITECTURES_H_

# include <iostream>
# include <cassert>

# include <cuimg/gpu/cuda.h>
# include <cuimg/obox2d.h>

namespace cuimg
{

  namespace arch
  {
    struct cpu
    {
      enum { l1_cache_size = 32 * 1024 };
    };
  }

  template <typename ARCH, typename D>
  static dim3 dimblock(ARCH, size_t elt_size, const D& domain)
  {
    dim3 res;
    if ((domain.ncols() * elt_size) > ARCH::l1_cache_size)
    {
      res.x = domain.ncols();
      res.y = ARCH::l1_cache_size / (domain.ncols() * elt_size);
      res.z = 1;
    }
    else
    {
      res.x = (ARCH::l1_cache_size / elt_size);
      res.y = 1;
      res.z = 1;
    }

    return res;
  }

}

#endif
