#ifndef CUIMG_HOST_IMAGE3D_HPP_
# define CUIMG_HOST_IMAGE3D_HPP_

# include <cuimg/cpu/host_image3d.h>
# include <cuimg/error.h>

namespace cuimg
{
  template <typename V>
  host_image3d<V>::host_image3d(unsigned nslices, unsigned nrows, unsigned ncols)
    : domain_(nslices, nrows, ncols),
      pitch_(ncols * sizeof(V))
  {
    data_ = boost::shared_ptr<V>(new V[nslices * nrows * ncols]());
  }

  template <typename V>
  host_image3d<V>::host_image3d(const domain_type& d)
    : domain_(d),
      pitch_(ncols() * sizeof(V))
  {
    data_ = boost::shared_ptr<V>(new V[d.nslices() * d.nrows() * d.ncols()]());
  }

  template <typename V>
  host_image3d<V>::host_image3d(const host_image3d<V>& img)
    : domain_(img.domain()),
      pitch_(img.pitch()),
      data_(img.data_)
  {
  }

  template <typename V>
  host_image3d<V>&
  host_image3d<V>::operator=(const host_image3d<V>& img)
  {
    domain_ = img.domain();
    pitch_ = img.pitch();
    data_ = img.data_;
    return *this;
  }

  template <typename V>
  const typename host_image3d<V>::domain_type& host_image3d<V>::domain() const
  {
    return domain_;
  }

  template <typename V>
  int host_image3d<V>::nslices() const
  {
    return domain_.nslices();
  }

  template <typename V>
  int host_image3d<V>::nrows() const
  {
    return domain_.nrows();
  }

  template <typename V>
  int host_image3d<V>::ncols() const
  {
    return domain_.ncols();
  }

  template <typename V>
  bool host_image3d<V>::has(const point& p) const
  {
    return domain_.has(p);
  }

  template <typename V>
  size_t host_image3d<V>::pitch() const
  {
    return pitch_;
  }

  template <typename V>
  V* host_image3d<V>::data()
  {
    return data_.get();
  }

  template <typename V>
  const V* host_image3d<V>::data() const
  {
    return data_.get();
  }

  template <typename V>
  V& host_image3d<V>::operator()(const point& p)
  {
    return data_.get()[p.sli() * domain_.ncols() * domain_.nrows()
                       + p.row() * domain_.ncols() + p.col()];
  }

  template <typename V>
  const V& host_image3d<V>::operator()(const point& p) const
  {
    return data_.get()[p.sli() * domain_.ncols() * domain_.nrows()
                       + p.row() * domain_.ncols() + p.col()];
  }

  template <typename V>
  V&
  host_image3d<V>::operator()(int s, int r, int c)
  {
    return operator()(point(s, r, c));
  }

  template <typename V>
  const V&
  host_image3d<V>::operator()(int s, int r, int c) const
  {
    return operator()(point(s, r, c));
  }

  template <typename V>
  V& host_image3d<V>::operator[](unsigned i)
  {
    return data_.get()[i];
  }

  template <typename V>
  const V& host_image3d<V>::operator[](unsigned i) const
  {
    return data_.get()[i];
  }

}

#endif
