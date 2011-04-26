#ifndef CUIMG_KERNEL_IMAGE2D_H_
# define CUIMG_KERNEL_IMAGE2D_H_

# include <cuda_runtime.h>
# include <cuimg/point2d.h>
# include <cuimg/gpu/image2d.h>

namespace cuimg
{
  template <typename V>
  class kernel_image2d
  {
  public:
    typedef V value_type;
    typedef point2d<int> point;
    typedef obox2d<point> domain_type;

    __host__ __device__ inline kernel_image2d();

    __host__ __device__ inline kernel_image2d(const kernel_image2d<V>& img);

    template <template <class> class PT>
    __host__ __device__ inline kernel_image2d(const image2d<V, PT>& img);

    __host__ __device__ inline kernel_image2d<V>& operator=(const kernel_image2d<V>& img);

    template <template <class> class PT>
    __host__ __device__ inline kernel_image2d<V>& operator=(const image2d<V, PT>& img);

    __host__ __device__ inline const domain_type& domain() const;
    __host__ __device__ inline unsigned nrows() const;
    __host__ __device__ inline unsigned ncols() const;
    __host__ __device__ inline unsigned pitch() const;

    __device__ inline V& operator()(const point& p);
    __device__ inline const V& operator()(const point& p) const;

    __device__ inline V& operator()(unsigned i, unsigned j);
    __device__ inline const V& operator()(unsigned i, unsigned j) const;

    __device__ inline V& operator[](unsigned i);
    __device__ inline const V& operator[](unsigned i) const;

    __host__ __device__ inline bool has(const point& p) const;

    __host__ __device__ inline V* data();
    __host__ __device__ inline const V* data() const;

  private:
    domain_type domain_;
    unsigned pitch_;
    V* data_;
  };

  template <typename V, template <class> class PT>
  kernel_image2d<V> mki(const image2d<V, PT>& img)
  {
    return img;
  }

}

# include <cuimg/gpu/kernel_image2d.hpp>

#endif
