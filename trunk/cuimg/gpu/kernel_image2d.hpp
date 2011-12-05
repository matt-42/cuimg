#ifndef KERNEL_IMAGE2D_HPP_
# define KERNEL_IMAGE2D_HPP_

# include <cuimg/point2d.h>
# include <cuimg/gpu/kernel_image2d.h>

namespace cuimg
{
  template <typename V>
  kernel_image2d<V>::kernel_image2d()
    : data_(0)
  {
  }

  template <typename V>
  kernel_image2d<V>::kernel_image2d(const kernel_image2d<V>& img)
    : domain_(img.domain()),
      pitch_(img.pitch()),
      data_(const_cast<V*>(img.data()))
  {
  }

  template <typename V>
  template <template <class> class PT>
  kernel_image2d<V>::kernel_image2d(const image2d<V, PT>& img)
    : domain_(img.domain()),
      pitch_(img.pitch()),
      data_(const_cast<V*>(img.data()))
  {
    assert(data_);
  }

  template <typename V>
  kernel_image2d<V>&
  kernel_image2d<V>::operator=(const kernel_image2d<V>& img)
  {
    domain_ = img.domain();
    pitch_ = img.pitch();
    data_ = img.data();
    return *this;
  }

  template <typename V>
  template <template <class> class PT>
  kernel_image2d<V>&
  kernel_image2d<V>::operator=(const image2d<V, PT>& img)
  {
    domain_ = img.domain();
    pitch_ = img.pitch();
    data_ = const_cast<V*>(img.data());
    assert(data_);
    return *this;
  }

  template <typename V>
  const typename kernel_image2d<V>::domain_type& kernel_image2d<V>::domain() const
  {
    return domain_;
  }

  template <typename V>
  unsigned kernel_image2d<V>::nrows() const
  {
    return domain_.nrows();
  }
  template <typename V>
  unsigned kernel_image2d<V>::ncols() const
  {
    return domain_.ncols();
  }

  template <typename V>
  unsigned kernel_image2d<V>::pitch() const
  {
    return pitch_;
  }

  template <typename V>
  V* kernel_image2d<V>::data()
  {
    return data_;
  }

  template <typename V>
  const V* kernel_image2d<V>::data() const
  {
    return data_;
  }

  template <typename V>
  bool kernel_image2d<V>::has(const point& p) const
  {
    return domain_.has(p);
  }


  template <typename V>
  V& kernel_image2d<V>::operator()(unsigned i, unsigned j)
  {
    return *(V*)(((char*)data_) + i * pitch_ + j * sizeof(V));
  }

  template <typename V>
  const V& kernel_image2d<V>::operator()(unsigned i, unsigned j) const
  {
    return *(V*)(((char*)data_) + i * pitch_ + j * sizeof(V));
  }

  template <typename V>
  V& kernel_image2d<V>::operator()(const point& p)
  {
    return *(V*)(((char*)data_) + p.row() * pitch_ + p.col() * sizeof(V));
  }

  template <typename V>
  const V& kernel_image2d<V>::operator()(const point& p) const
  {
    return *(V*)(((char*)data_) + p.row() * pitch_ + p.col() * sizeof(V));
  }

  template <typename V>
  V& kernel_image2d<V>::operator[](unsigned i)
  {
    return data_[i];
  }

  template <typename V>
  const V& kernel_image2d<V>::operator[](unsigned i) const
  {
    return data_[i];
  }

}

#endif
