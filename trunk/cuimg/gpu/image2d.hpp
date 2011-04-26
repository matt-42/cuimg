#ifndef CUIMG_IMAGE2D_HPP_
# define CUIMG_IMAGE2D_HPP_

# include <cuimg/gpu/image2d.h>
# include <cuimg/error.h>

namespace cuimg
{
  template <typename V, template <class> class PT>
  image2d<V, PT>::image2d(unsigned nrows, unsigned ncols)
    : domain_(nrows, ncols)
  {
    V* b;
    cudaMallocPitch((void**)&b, &pitch_, domain_.ncols() * sizeof(V), domain_.nrows());
    check_cuda_error();
    assert(b);
    data_ = PT<V>(b, cudaFree);
    data_ptr_ = data_.get();
  }

  template <typename V, template <class> class PT>
  image2d<V, PT>::image2d(V* data, unsigned nrows, unsigned ncols, unsigned pitch)
    : domain_(nrows, ncols),
      pitch_(pitch),
      data_(data),
      data_ptr_(data)
  {
  }

  template <typename V, template <class> class PT>
  image2d<V, PT>::image2d(const domain_type& d)
    : domain_(d)
  {
    V* b;
    cudaMallocPitch((void**)&b, &pitch_, domain_.ncols() * sizeof(V), domain_.nrows());
    check_cuda_error();
    assert(b);
    data_ = PT<V>(b, cudaFree);
    data_ptr_ = data_.get();
  }

  template <typename V, template <class> class PT>
  template <template <class> class IPT>
  image2d<V, PT>::image2d(const image2d<V, IPT>& img)
    : domain_(img.domain()),
      pitch_(img.pitch()),
      data_(img.data_sptr()),
      data_ptr_(const_cast<V*>(img.data()))
  {
  }

  template <typename V, template <class> class PT>
  template <template <class> class IPT>
  image2d<V, PT>&
  image2d<V, PT>::operator=(const image2d<V, IPT>& img)
  {
    domain_ = img.domain();
    pitch_ = img.pitch();
    data_ = img.data_sptr();
    data_ptr_ = img.data();
    return *this;
  }

  template <typename V, template <class> class PT>
  const typename image2d<V, PT>::domain_type& image2d<V, PT>::domain() const
  {
    return domain_;
  }

  template <typename V, template <class> class PT>
  unsigned image2d<V, PT>::nrows() const
  {
    return domain_.nrows();
  }
  template <typename V, template <class> class PT>
  unsigned image2d<V, PT>::ncols() const
  {
    return domain_.ncols();
  }

  template <typename V, template <class> class PT>
  size_t image2d<V, PT>::pitch() const
  {
    return pitch_;
  }

  template <typename V, template <class> class PT>
  V* image2d<V, PT>::data()
  {
    return data_ptr_;
  }

  template <typename V, template <class> class PT>
  const V* image2d<V, PT>::data() const
  {
    return data_ptr_;
  }

  template <typename V, template <class> class PT>
  bool image2d<V, PT>::has(const point& p) const
  {
    return domain_.has(p);
  }

  template <typename V, template <class> class PT>
  const PT<V>
  image2d<V, PT>::data_sptr() const
  {
    return data_;
  }

  template <typename V, template <class> class PT>
  PT<V>
  image2d<V, PT>::data_sptr()
  {
    return data_;
  }

}

#endif
