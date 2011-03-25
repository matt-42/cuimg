#ifndef CUIMG_IMAGE2D_HPP_
# define CUIMG_IMAGE2D_HPP_

# include <cuimg/gpu/image2d.h>
# include <cuimg/error.h>

namespace cuimg
{
  template <typename V>
  image2d<V>::image2d(unsigned nrows, unsigned ncols)
    : domain_(nrows, ncols)
  {
    V* b;
    cudaMallocPitch((void**)&b, &pitch_, domain_.ncols() * sizeof(V), domain_.nrows());
    check_cuda_error();
    assert(b);
    data_ = boost::shared_ptr<V>(b, cudaFree);
    data_ptr_ = data_.get();
  }

  template <typename V>
  image2d<V>::image2d(const domain_type& d)
    : domain_(d)
  {
    V* b;
    cudaMallocPitch((void**)&b, &pitch_, domain_.ncols() * sizeof(V), domain_.nrows());
    check_cuda_error();
    assert(b);
    data_ = boost::shared_ptr<V>(b, cudaFree);
    data_ptr_ = data_.get();
  }

  template <typename V>
  image2d<V>::image2d(const image2d<V>& img)
    : domain_(img.domain()),
      pitch_(img.pitch()),
      data_(img.data_),
      data_ptr_(img.data_ptr_)
  {
  }

  template <typename V>
  image2d<V>&
  image2d<V>::operator=(const image2d<V>& img)
  {
    domain_ = img.domain();
    pitch_ = img.pitch();
    data_ = img.data_;
    data_ptr_ = img.data_ptr_;
    return *this;
  }

  template <typename V>
  const typename image2d<V>::domain_type& image2d<V>::domain() const
  {
    return domain_;
  }

  template <typename V>
  unsigned image2d<V>::nrows() const
  {
    return domain_.nrows();
  }
  template <typename V>
  unsigned image2d<V>::ncols() const
  {
    return domain_.ncols();
  }

  template <typename V>
  size_t image2d<V>::pitch() const
  {
    return pitch_;
  }

  template <typename V>
  V* image2d<V>::data()
  {
    return data_ptr_;
  }

  template <typename V>
  const V* image2d<V>::data() const
  {
    return data_ptr_;
  }

  template <typename V>
  bool image2d<V>::has(const point& p) const
  {
    return domain_.has(p);
  }

  template <typename V>
  const boost::shared_ptr<V>
  image2d<V>::data_sptr() const
  {
    return data_;
  }

  template <typename V>
  boost::shared_ptr<V>
  image2d<V>::data_sptr()
  {
    return data_;
  }

}

#endif
