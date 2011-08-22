#ifndef CUIMG_KERNEL_IMAGE3D_H_
# define CUIMG_KERNEL_IMAGE3D_H_

# include <cuda_runtime.h>
# include <cuimg/point3d.h>
# include <cuimg/gpu/image3d.h>

namespace cuimg
{
  template <typename V>
  class kernel_image3d
  {
  public:
    typedef V value_type;
    typedef point3d<int> point;
    typedef obox3d<point> domain_type;

    __host__ __device__ inline kernel_image3d(const kernel_image3d<V>& img);
    __host__ __device__ inline kernel_image3d(const image3d<V>& img);

    __host__ __device__ inline kernel_image3d<V>& operator=(const kernel_image3d<V>& img);
    __host__ __device__ inline kernel_image3d<V>& operator=(const image3d<V>& img);

    __host__ __device__ inline const domain_type& domain() const;
    __host__ __device__ inline unsigned nrows() const;
    __host__ __device__ inline unsigned ncols() const;
    __host__ __device__ inline unsigned nslices() const;
    __host__ __device__ inline unsigned pitch() const;

    __device__ inline V& operator()(const point& p);
    __device__ inline const V& operator()(const point& p) const;

    __device__ inline V& operator[](unsigned i);
    __device__ inline const V& operator[](unsigned i) const;

    __device__ inline V& operator()(unsigned s, unsigned r, unsigned c);
    __device__ inline const V& operator()(unsigned s, unsigned r, unsigned c) const;

    __device__ inline V& operator()(unsigned s, const point2d<int>& p);
    __device__ inline const V& operator()(unsigned s, const point2d<int>& p) const;

    __host__ __device__ inline bool has(const point& p) const;

    __host__ __device__ inline V* data();
    __host__ __device__ inline const V* data() const;

  private:
    domain_type domain_;
    unsigned pitch_;
    V* data_;
  };

  template <typename V>
  kernel_image3d<V> mki(const image3d<V>& img)
  {
    return img;
  }

}

# include <cuimg/gpu/kernel_image3d.hpp>

#endif
