// -*- mode: cuda-mode -*-

#ifndef CUIMG_IMAGE2D_H_
# define CUIMG_IMAGE2D_H_

# include <cuimg/gpu/cuda.h>
# include <boost/shared_ptr.hpp>
# include <cuimg/target.h>
# include <cuimg/concepts.h>
# include <cuimg/point2d.h>
# include <cuimg/obox2d.h>
# include <cuimg/dsl/expr.h>

namespace cuimg
{
  template <typename V>
  class device_image2d : public Image2d<device_image2d<V> >
  {
  public:
    enum { target = GPU };

    typedef boost::shared_ptr<V> PT;
    typedef V value_type;
    typedef point2d<int> point;
    typedef obox2d<point> domain_type;

    inline device_image2d();
    inline device_image2d(unsigned nrows, unsigned ncols);
    inline device_image2d(V* data, unsigned nrows, unsigned ncols, unsigned pitch);
    inline device_image2d(const domain_type& d);

    __host__ __device__ inline device_image2d(const device_image2d<V>& d);

    __host__ __device__ inline device_image2d<V>& operator=(const device_image2d<V>& d);

    template <typename E>
    __host__ inline device_image2d<V>& operator=(const expr<E>& e);

    __host__ __device__ inline const domain_type& domain() const;
    __host__ __device__ inline unsigned nrows() const;
    __host__ __device__ inline unsigned ncols() const;
    __host__ __device__ inline size_t pitch() const;

    __host__ inline V read_back_pixel(const point& p) const;
    __host__ inline void set_pixel(const point& p, const V& e);

    __host__ __device__ inline V* data() const;
    //__host__ __device__ inline const V* data() const;

    __host__ __device__ inline bool has(const point& p) const;

    __host__ __device__ inline const PT data_sptr() const;
    __host__ __device__ inline PT data_sptr();

  private:
    domain_type domain_;
    size_t pitch_;
    PT data_;
    V* data_ptr_;
  };

}

# include <cuimg/gpu/kernel_image2d.h>

# include <cuimg/gpu/device_image2d.hpp>

#endif
