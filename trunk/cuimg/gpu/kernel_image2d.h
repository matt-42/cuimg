#ifndef CUIMG_KERNEL_IMAGE2D_H_
# define CUIMG_KERNEL_IMAGE2D_H_

# include <cuda_runtime.h>
# include <cuimg/point2d.h>
# include <cuimg/concepts.h>
# include <cuimg/gpu/device_image2d.h>
# include <cuimg/cpu/host_image2d.h>

namespace cuimg
{
  template <typename V>
  class kernel_image2d : public Image2d<kernel_image2d<V> >
  {
  public:
    typedef V value_type;
    typedef point2d<int> point;
    typedef obox2d<point> domain_type;

    __host__ __device__ inline kernel_image2d();

    __host__ __device__ inline kernel_image2d(const kernel_image2d<V>& img);

    template <typename I>
    __host__ __device__ inline kernel_image2d(const Image2d<I>& img);

    __host__ __device__ inline kernel_image2d<V>& operator=(const kernel_image2d<V>& img);

    template <typename I>
    __host__ __device__ inline kernel_image2d<V>& operator=(const Image2d<I>& img);

    __host__ __device__ inline const domain_type& domain() const;
    __host__ __device__ inline unsigned nrows() const;
    __host__ __device__ inline unsigned ncols() const;
    __host__ __device__ inline unsigned pitch() const;

    __host__ __device__ inline V& operator()(const point& p);
    __host__ __device__ inline const V& operator()(const point& p) const;

    __host__ __device__ inline V& operator()(unsigned i, unsigned j);
    __host__ __device__ inline const V& operator()(unsigned i, unsigned j) const;

    __host__ __device__ inline V& operator[](unsigned i);
    __host__ __device__ inline const V& operator[](unsigned i) const;

    __host__ __device__ inline bool has(const point& p) const;

    __host__ __device__ inline V* data();
    __host__ __device__ inline const V* data() const;

  private:
    domain_type domain_;
    unsigned pitch_;
    V* data_;
  };

  template <typename I>
  kernel_image2d<typename I::value_type> mki(const Image2d<I>& img)
  {
    assert(exact(img).data());
    return exact(img);
  }

}

# include <cuimg/gpu/kernel_image2d.hpp>

#endif
