#ifndef CUIMG_KERNEL_IMAGE2D_H_
# define CUIMG_KERNEL_IMAGE2D_H_

# include <cuimg/gpu/cuda.h>
# include <cuimg/point2d.h>
# include <cuimg/concepts.h>
# include <cuimg/gpu/device_image2d.h>
# include <cuimg/cpu/host_image2d.h>

#ifndef __GNUC__
# define BOOST_TYPEOF_COMPLIANT
#endif
# include <boost/typeof/typeof.hpp>
# include BOOST_TYPEOF_INCREMENT_REGISTRATION_GROUP()

namespace cuimg
{
  template <typename V>
  class kernel_image2d : public Image2d<kernel_image2d<V> >
  {
  public:
    typedef int is_expr;
    typedef V value_type;
    typedef point2d<int> point;
    typedef obox2d domain_type;

    __host__ __device__ inline kernel_image2d();

    __host__ __device__ inline kernel_image2d(const kernel_image2d<V>& img);

    template <typename I>
    __host__ __device__ inline kernel_image2d(const Image2d<I>& img);

    __host__ __device__ inline kernel_image2d<V>& operator=(const kernel_image2d<V>& img);

    template <typename I>
    __host__ __device__ inline kernel_image2d<V>& operator=(const Image2d<I>& img);

    __host__ __device__ inline const domain_type& domain() const;
    __host__ __device__ inline int nrows() const;
    __host__ __device__ inline int ncols() const;
    __host__ __device__ inline int pitch() const;

    __host__ __device__ inline V& operator()(const point& p);
    __host__ __device__ inline const V& operator()(const point& p) const;

    __host__ __device__ inline V& eval(const point& p)             { return (*this)(p); };
    __host__ __device__ inline const V& eval(const point& p) const { return (*this)(p); };

    __host__ __device__ inline V& operator()(int i, int j);
    __host__ __device__ inline const V& operator()(int i, int j) const;

    __host__ __device__ inline V& operator[](int i);
    __host__ __device__ inline const V& operator[](int i) const;

    __host__ __device__ inline bool has(const point& p) const;

    __host__ __device__ inline V* data();
    __host__ __device__ inline const V* data() const;

    __host__ __device__ inline V* begin() const;
    __host__ __device__ inline V* end() const;

  private:
    domain_type domain_;
    int pitch_;
    V* data_;
  };


  template <typename T>
  struct return_type;

  template <typename V>
  struct return_type<kernel_image2d<V> >
  {
    typedef typename kernel_image2d<V>::value_type ret;
  };

  template <typename I>
  kernel_image2d<typename I::value_type> mki(const Image2d<I>& img)
  {
    assert(exact(img).data());
    return exact(img);
  }

}

BOOST_TYPEOF_REGISTER_TEMPLATE(cuimg::kernel_image2d, (typename))

# include <cuimg/gpu/kernel_image2d.hpp>

#endif
