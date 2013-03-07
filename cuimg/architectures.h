#ifndef CUIMG_ARCHITECTURES_H_
# define CUIMG_ARCHITECTURES_H_

# include <iostream>
# include <cassert>

# include <cuimg/gpu/cuda.h>
# include <cuimg/obox2d.h>
# include <cuimg/util.h>


#ifndef NO_CUDA
  #include <thrust/device_vector.h>
#endif

namespace cuimg
{

  template <typename V> struct host_image2d;
  template <typename V> struct device_image2d;

  struct cpu
  {
    template <typename V> struct image2d { typedef host_image2d<V> ret; };
    template <typename V> struct kernel_image2d { typedef host_image2d<V> ret; };
    template <typename V> struct vector { typedef std::vector<V> ret; };

    enum { l1_cache_size = 32 * 1024 };
  };

#ifndef NO_CUDA
  struct cuda_gpu
  {
    template <typename V> struct image2d { typedef device_image2d<V> ret; };
    template <typename V> struct kernel_image2d { typedef kernel_image2d<V> ret; };
    template <typename V> struct vector { typedef thrust::device_vector<V> ret; };

    dim3 dimblock1d() {             return dim3(256, 1, 1); }
    dim3 dimgrid1d(unsigned size) { return dim3(idivup(size, 256), 1, 1); }

    dim3 dimblock2d() { return dim3(16, 16, 1); }
    dim3 dimgrid2d(const obox2d& d) { return dim3(idivup(d.ncols(), 16), idivup(d.nrows(), 16), 1); }
  };
#endif

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
