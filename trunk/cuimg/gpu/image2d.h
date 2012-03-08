// -*- mode: cuda-mode -*-

#ifndef CUIMG_IMAGE2D_H_
# define CUIMG_IMAGE2D_H_

# include <boost/shared_ptr.hpp>
# include <cuimg/target.h>
# include <cuimg/concepts.h>
# include <cuimg/point2d.h>
# include <cuimg/obox2d.h>
# include <cuimg/dsl/expr.h>

namespace cuimg
{
  template <typename V>
  class image2d : public Image2d<image2d<V> >
  {
  public:
    enum { target = GPU };

    typedef boost::shared_ptr<V> PT;
    typedef V value_type;
    typedef point2d<int> point;
    typedef obox2d<point> domain_type;

    inline image2d();
    inline image2d(unsigned nrows, unsigned ncols);
    inline image2d(V* data, unsigned nrows, unsigned ncols, unsigned pitch);
    inline image2d(const domain_type& d);

    __host__ __device__ inline image2d(const image2d<V>& d);

    __host__ __device__ inline image2d<V>& operator=(const image2d<V>& d);

    template <typename E>
    __host__ inline image2d<V>& operator=(const expr<E>& e);

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

# include <cuimg/gpu/image2d.hpp>

#endif
