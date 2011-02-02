#ifndef CUIMG_HOST_IMAGE2D_HPP_
# define CUIMG_HOST_IMAGE2D_HPP_

# include <cuimg/host_image2d.h>
# include <cuimg/error.h>

namespace cuimg
{
  template <typename V>
  host_image2d<V>::host_image2d(unsigned nrows, unsigned ncols)
    : domain_(nrows, ncols),
      pitch_(ncols * sizeof(V))
  {
    data_ = boost::shared_ptr<V>(new V[nrows * ncols]());
  }

  template <typename V>
  host_image2d<V>::host_image2d(const domain_type& d)
    : domain_(d),
      pitch_(ncols() * sizeof(V))
  {
    data_ = boost::shared_ptr<V>(new V[d.nrows() * d.ncols()]());
  }

  template <typename V>
  host_image2d<V>::host_image2d(const host_image2d<V>& img)
    : domain_(img.domain()),
      pitch_(img.pitch()),
      data_(img.data_)
  {
  }

  template <typename V>
  host_image2d<V>&
  host_image2d<V>::operator=(const host_image2d<V>& img)
  {
    domain_ = img.domain();
    pitch_ = img.pitch();
    data_ = img.data_;
    return *this;
  }

  template <typename V>
  const typename host_image2d<V>::domain_type& host_image2d<V>::domain() const
  {
    return domain_;
  }

  template <typename V>
  unsigned host_image2d<V>::nrows() const
  {
    return domain_.nrows();
  }
  template <typename V>
  unsigned host_image2d<V>::ncols() const
  {
    return domain_.ncols();
  }

  template <typename V>
  bool host_image2d<V>::has(const point& p) const
  {
    return domain_.has(p);
  }

  template <typename V>
  size_t host_image2d<V>::pitch() const
  {
    return pitch_;
  }

  template <typename V>
  V* host_image2d<V>::data()
  {
    return data_.get();
  }

  template <typename V>
  const V* host_image2d<V>::data() const
  {
    return data_.get();
  }

  template <typename V>
  V& host_image2d<V>::operator()(const point& p)
  {
    return data_.get()[p.row() * domain_.ncols() + p.col()];
  }

  template <typename V>
  const V& host_image2d<V>::operator()(const point& p) const
  {
    return data_.get()[p.row() * domain_.ncols() + p.col()];
  }

  template <typename V>
  V&
  host_image2d<V>::operator()(int r, int c)
  {
    return operator()(point(r, c));
  }

  template <typename V>
  const V&
  host_image2d<V>::operator()(int r, int c) const
  {
    return operator()(point(r, c));
  }

}

#endif
