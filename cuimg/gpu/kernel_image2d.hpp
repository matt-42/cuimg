#ifndef KERNEL_IMAGE2D_HPP_
# define KERNEL_IMAGE2D_HPP_

# include <cassert>
# include <cuimg/point2d.h>
# include <cuimg/gpu/kernel_image2d.h>

namespace cuimg
{
  template <typename V>
  kernel_image2d<V>::kernel_image2d()
    : data_(0)
  {
  }


  inline __host__ __device__ void my_assert(bool b)
  {
    assert(b);
  }

  template <typename V>
  kernel_image2d<V>::kernel_image2d(const kernel_image2d<V>& img)
    : domain_(img.domain()),
      pitch_(img.pitch()),
      data_(const_cast<V*>(img.data()))
  {
  }

  template <typename V>
  template <typename I>
  kernel_image2d<V>::kernel_image2d(const Image2d<I>& img)
    : domain_(exact(img).domain()),
      pitch_(exact(img).pitch()),
      data_((V*)(exact(img).data()))
  {
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
  template <typename I>
  kernel_image2d<V>&
  kernel_image2d<V>::operator=(const Image2d<I>& img_)
  {
    const I& img = exact(img_);
    domain_ = img.domain();
    pitch_ = img.pitch();
    data_ = const_cast<V*>(img.data());
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
  V* kernel_image2d<V>::begin() const
  {
    return data_;
  }

  template <typename V>
  V* kernel_image2d<V>::end() const
  {
    return data_ + (pitch_ * nrows()) / sizeof(V);
  }

  template <typename V>
  bool kernel_image2d<V>::has(const point& p) const
  {
    return domain_.has(p);
  }


  template <typename V>
  V& kernel_image2d<V>::operator()(unsigned i, unsigned j)
  {
    my_assert(data_);
    return *(V*)(((char*)data_) + i * pitch_ + j * sizeof(V));
  }

  template <typename V>
  const V& kernel_image2d<V>::operator()(unsigned i, unsigned j) const
  {
    my_assert(data_);
    return *(V*)(((char*)data_) + i * pitch_ + j * sizeof(V));
  }

  template <typename V>
  V& kernel_image2d<V>::operator()(const point& p)
  {
    my_assert(data_);
    return *(V*)(((char*)data_) + p.row() * pitch_ + p.col() * sizeof(V));
  }

  template <typename V>
  const V& kernel_image2d<V>::operator()(const point& p) const
  {
    my_assert(data_);
    return *(V*)(((char*)data_) + p.row() * pitch_ + p.col() * sizeof(V));
  }

  template <typename V>
  V& kernel_image2d<V>::operator[](unsigned i)
  {
    my_assert(data_);
    return data_[i];
  }

  template <typename V>
  const V& kernel_image2d<V>::operator[](unsigned i) const
  {
    my_assert(data_);
    return data_[i];
  }

}

#endif
