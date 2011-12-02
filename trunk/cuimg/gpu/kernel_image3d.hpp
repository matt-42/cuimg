#ifndef KERNEL_IMAGE3D_HPP_
# define KERNEL_IMAGE3D_HPP_

# include <cuimg/point3d.h>
# include <cuimg/gpu/kernel_image3d.h>

namespace cuimg
{

  template <typename V>
  kernel_image3d<V>::kernel_image3d()
    : data_(0)
  {
  }

  template <typename V>
  kernel_image3d<V>::kernel_image3d(const kernel_image3d<V>& img)
    : domain_(img.domain()),
      pitch_(img.pitch()),
      data_(const_cast<V*>(img.data()))
  {
  }

  template <typename V>
  kernel_image3d<V>::kernel_image3d(const image3d<V>& img)
    : domain_(img.domain()),
      pitch_(img.pitch()),
      data_(const_cast<V*>(img.data()))
  {
    assert(data_);
  }

  template <typename V>
  kernel_image3d<V>&
  kernel_image3d<V>::operator=(const kernel_image3d<V>& img)
  {
    domain_ = img.domain();
    pitch_ = img.pitch();
    data_ = img.data();
    assert(data_);
    return *this;
  }

  template <typename V>
  kernel_image3d<V>&
  kernel_image3d<V>::operator=(const image3d<V>& img)
  {
    domain_ = img.domain();
    pitch_ = img.pitch();
    data_ = img.data();
    return *this;
  }

  template <typename V>
  const typename kernel_image3d<V>::domain_type& kernel_image3d<V>::domain() const
  {
    return domain_;
  }

  template <typename V>
  unsigned kernel_image3d<V>::nslices() const
  {
    return domain_.nslices();
  }

  template <typename V>
  unsigned kernel_image3d<V>::nrows() const
  {
    return domain_.nrows();
  }
  template <typename V>
  unsigned kernel_image3d<V>::ncols() const
  {
    return domain_.ncols();
  }

  template <typename V>
  unsigned kernel_image3d<V>::pitch() const
  {
    return pitch_;
  }

  template <typename V>
  V* kernel_image3d<V>::data()
  {
    return data_;
  }

  template <typename V>
  const V* kernel_image3d<V>::data() const
  {
    return data_;
  }

  template <typename V>
  bool kernel_image3d<V>::has(const point& p) const
  {
    return domain_.has(p);
  }


  template <typename V>
  V& kernel_image3d<V>::operator()(unsigned s, unsigned r, unsigned c)
  {
    return *(V*)(((char*)data_) + s * pitch_ * nrows() + r * pitch_ + c * sizeof(V));
  }

  template <typename V>
  const V& kernel_image3d<V>::operator()(unsigned s, unsigned r, unsigned c) const
  {
    return *(V*)(((char*)data_) + s * pitch_ * nrows() + r * pitch_ + c * sizeof(V));
  }


    template <typename V>
  V& kernel_image3d<V>::operator()(unsigned s, const point2d<int>& p)
  {
    return *(V*)(((char*)data_) + s * pitch_ * nrows() + p.row() * pitch_ + p.col() * sizeof(V));
  }

  template <typename V>
  const V& kernel_image3d<V>::operator()(unsigned s, const point2d<int>& p) const
  {
    return *(V*)(((char*)data_) + s * pitch_ * nrows() + p.row() * pitch_ + p.col() * sizeof(V));
  }


  template <typename V>
  V& kernel_image3d<V>::operator()(const point& p)
  {
    return *(V*)(((char*)data_) + p.sli() * pitch_ * nrows() + p.row() * pitch_ + p.col() * sizeof(V));
  }

  template <typename V>
  const V& kernel_image3d<V>::operator()(const point& p) const
  {
    return *(V*)(((char*)data_) + p.sli() * pitch_ * nrows() + p.row() * pitch_ + p.col() * sizeof(V));
  }

  template <typename V>
  V& kernel_image3d<V>::operator[](unsigned i)
  {
    return data_[i];
  }

  template <typename V>
  const V& kernel_image3d<V>::operator[](unsigned i) const
  {
    return data_[i];
  }

}

#endif
