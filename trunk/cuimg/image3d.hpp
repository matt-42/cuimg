#ifndef CUIMG_IMAGE3D_HPP_
# define CUIMG_IMAGE3D_HPP_

# include <cuimg/image3d.h>
# include <cuimg/error.h>

namespace cuimg
{
  template <typename V>
  image3d<V>::image3d(unsigned nslices, unsigned nrows, unsigned ncols)
    : domain_(nslices, nrows, ncols)
  {
    cudaExtent extent;
    extent.width  = ncols * sizeof(V);
    extent.height = nrows;
    extent.depth  = nslices;

    cudaPitchedPtr pp;
    cudaMalloc3D(&pp, extent);
    check_cuda_error();
    assert(pp.ptr);
    data_ = boost::shared_ptr<V>((V*)pp.ptr, cudaFree);
    pitch_ = pp.pitch;
    data_ptr_ = data_.get();
  }

  template <typename V>
  image3d<V>::image3d(const domain_type& d)
    : domain_(d)
  {
    cudaExtent extent;
    extent.width  = domain_.ncols() * sizeof(V);
    extent.height = domain_.nrows();
    extent.depth  = domain_.nslices();

    cudaPitchedPtr pp;
    cudaMalloc3D(&pp, extent);
    check_cuda_error();
    assert(pp.ptr);
    data_ = boost::shared_ptr<V>((V*)pp.ptr, cudaFree);
    pitch_ = pp.pitch;
    data_ptr_ = data_.get();
  }

  template <typename V>
  image3d<V>::image3d(const image3d<V>& img)
    : domain_(img.domain()),
      pitch_(img.pitch()),
      data_(img.data_),
      data_ptr_(img.data_ptr_)
  {
  }

  template <typename V>
  image3d<V>&
  image3d<V>::operator=(const image3d<V>& img)
  {
    domain_ = img.domain();
    pitch_ = img.pitch();
    data_ = img.data_;
    data_ptr_ = img.data_ptr_;
    return *this;
  }

  template <typename V>
  const typename image3d<V>::domain_type& image3d<V>::domain() const
  {
    return domain_;
  }

  template <typename V>
  unsigned image3d<V>::nrows() const
  {
    return domain_.nrows();
  }

  template <typename V>
  unsigned image3d<V>::nslices() const
  {
    return domain_.nslices();
  }
  template <typename V>
  unsigned image3d<V>::ncols() const
  {
    return domain_.ncols();
  }

  template <typename V>
  size_t image3d<V>::pitch() const
  {
    return pitch_;
  }

  template <typename V>
  V* image3d<V>::data()
  {
    return data_ptr_;
  }

  template <typename V>
  const V* image3d<V>::data() const
  {
    return data_ptr_;
  }
}

#endif
