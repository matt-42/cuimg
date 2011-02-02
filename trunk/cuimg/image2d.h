// -*- mode: cuda-mode -*-

#ifndef CUIMG_IMAGE2D_H_
# define CUIMG_IMAGE2D_H_

# include <boost/shared_ptr.hpp>
# include <cuimg/point2d.h>
# include <cuimg/obox2d.h>

namespace cuimg
{
  template <typename V>
  class image2d
  {
  public:
    typedef point2d<int> point;
    typedef obox2d<point> domain_type;

    inline image2d(unsigned nrows, unsigned ncols);
    inline image2d(const domain_type& d);

    __host__ __device__ inline image2d(const image2d<V>& d);
    __host__ __device__ inline image2d<V>& operator=(const image2d<V>& d);

    __host__ __device__ inline const domain_type& domain() const;
    __host__ __device__ inline unsigned nrows() const;
    __host__ __device__ inline unsigned ncols() const;
    __host__ __device__ inline size_t pitch() const;

    __host__ __device__ inline V* data();
    __host__ __device__ inline const V* data() const;

    __host__ __device__ inline bool has(const point& p) const;

    __host__ inline const boost::shared_ptr<V> data_sptr() const;
    __host__ inline boost::shared_ptr<V> data_sptr();

  private:
    domain_type domain_;
    size_t pitch_;
    boost::shared_ptr<V> data_;
    V* data_ptr_;
  };

}

# include <cuimg/image2d.hpp>

#endif
