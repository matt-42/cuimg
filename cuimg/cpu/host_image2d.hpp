#ifndef CUIMG_HOST_IMAGE2D_HPP_
# define CUIMG_HOST_IMAGE2D_HPP_

# include <cuimg/error.h>
# include <cuimg/free.h>

#ifdef WITH_OPENCV
#include <opencv2/core/core.hpp>
#endif

namespace cuimg
{


  template <typename V>
  host_image2d<V>::host_image2d()
    : buffer_(0)
  {
  }

  template <typename V>
  host_image2d<V>::host_image2d(unsigned nrows, unsigned ncols, bool pinned)
    : domain_(nrows, ncols)
  {
    allocate(domain_, pinned);
  }

  template <typename V>
  host_image2d<V>::host_image2d(V* data, unsigned nrows, unsigned ncols, unsigned pitch)
    : domain_(nrows, ncols),
      pitch_(pitch)
  {
    data_ = PT(data, dummy_free<V>);
    buffer_ = data_.get();
  }

  template <typename V>
  host_image2d<V>::host_image2d(const domain_type& d, bool pinned)
    : domain_(d),
      pitch_(ncols() * sizeof(V))
  {
    allocate(domain_, pinned);
  }

  template <typename V>
  void
  host_image2d<V>::allocate(const domain_type& d, bool pinned)
  {
    V* ptr;

#ifndef NO_CUDA
    if (pinned)
    {
      cudaMallocHost(&ptr, domain_.nrows() * domain_.ncols() * sizeof(V));
      pitch_ = domain_.ncols() * sizeof(V);
      data_ = boost::shared_ptr<V>(ptr, cudaFreeHost);
      buffer_ = data_.get();
    }
    else
#endif
    {
      pitch_ = d.ncols() * sizeof(V);
      if (pitch_ % 4)
      	pitch_ = pitch_ + 4 - (pitch_ & 3);
      ptr = (V*) new char[domain_.nrows() * pitch_ + 64];
      data_ = boost::shared_ptr<V>(ptr, array_free<V>);

      buffer_ = data_.get();
      if (size_t(buffer_) % 4)
      	buffer_ = (V*)((size_t(buffer_) + 4 - (size_t(buffer_) & 3)   ));

      // assert(!(size_t(buffer_) % 64));
      // assert(!(size_t(pitch_) % 64));
    }

    assert(buffer_);
  }
//   template <typename V>
//   void
//   host_image2d<V>::allocate(const domain_type& d, bool pinned)
//   {
//     V* ptr;

// #ifndef NO_CUDA
//     if (pinned)
//     {
//       cudaMallocHost(&ptr, domain_.nrows() * domain_.ncols() * sizeof(V));
//       data_ = boost::shared_ptr<V>(ptr, cudaFreeHost);
//     }
//     else
// #endif
//     {
//       ptr = new V[domain_.nrows() * domain_.ncols()];
//       data_ = boost::shared_ptr<V>(ptr, array_free<V>);
//     }

//     buffer_ = data_.get();

//     buffer_ = data_.get();
//   }



#ifdef WITH_OPENCV
  template <typename V>
  host_image2d<V>::host_image2d(IplImage* imgIpl)
  {
    *this = imgIpl;
  }

  template <typename V>
  host_image2d<V>::host_image2d(cv::Mat m)
  {
    m.addref();
    pitch_ = m.step;
    data_ = PT((V*) m.data, dummy_free<V>);
    buffer_ = (V*) m.data;
    domain_ = domain_type(m.rows, m.cols);
    // *this = static_cast<IplImage*>(&m);
  }
#endif

  template <typename V>
  host_image2d<V>&
  host_image2d<V>::operator=(const host_image2d<V>& img)
  {
    domain_ = img.domain();
    pitch_ = img.pitch();
    data_ = img.data_;
    buffer_ = img.buffer_;
    return *this;
  }

  template <typename V>
  host_image2d<V>::host_image2d(const host_image2d<V>& img)
    : domain_(img.domain()),
      pitch_(img.pitch()),
      data_(img.data_)
  {
    buffer_ = img.buffer_;
  }

#ifdef WITH_OPENCV
  template <typename V>
  host_image2d<V>&
  host_image2d<V>::operator=(IplImage *imgIpl)
  {
    pitch_ = imgIpl->widthStep;
    data_ = PT((V*) imgIpl->imageData, dummy_free<V>);
    buffer_ = (V*) imgIpl->imageData;
    domain_ = domain_type(imgIpl->height,imgIpl->width);
    return *this;
  }


  template <typename V>
  host_image2d<V>&
  host_image2d<V>::operator=(cv::Mat m)
  {
    m.addref();
    pitch_ = m.step;
    data_ = PT((V*) m.data, dummy_free<V>);
    buffer_ = (V*) m.data;
    domain_ = domain_type(m.rows, m.cols);
  }

#endif

  template <typename V>
  const typename host_image2d<V>::domain_type& host_image2d<V>::domain() const
  {
    assert(buffer_);
    return domain_;
  }

  template <typename V>
  int host_image2d<V>::nrows() const
  {
    return domain_.nrows();
  }
  template <typename V>
  int host_image2d<V>::ncols() const
  {
    return domain_.ncols();
  }

  template <typename V>
  bool host_image2d<V>::has(const point& p) const
  {
    return domain_.has(p);
  }

  template <typename V>
  inline size_t host_image2d<V>::pitch() const
  {
    return pitch_;
  }


#ifdef WITH_OPENCV
  //  Get IplImage from host_image2d --
  template <typename V>
  IplImage* host_image2d<V>::getIplImage() const
  {
    assert(buffer_);
    //allocate the structure
    IplImage* frameIPL = cvCreateImageHeader(cvSize(ncols(),nrows()),
					     sizeof(typename V::vtype)*8,
					     V::size);
    //init the data structure
    cvSetData(frameIPL, (void*)data(), pitch());
    return frameIPL;
  }

  template <typename V>
  host_image2d<V>::operator cv::Mat() const
  {
    assert(buffer_);
    return cv::Mat(nrows(), ncols(), sizeof(typename V::vtype)*8, (void*)data(), pitch());
  }

#endif
  template <typename V>
  inline V* host_image2d<V>::data()
  {
    return buffer_;
  }

  template <typename V>
  inline const V* host_image2d<V>::data() const
  {
    return buffer_;
  }

  template <typename V>
  inline size_t host_image2d<V>::buffer_size() const
  {
    assert(buffer_);
    return nrows() * pitch_;
  }

  template <typename V>
  inline V& host_image2d<V>::operator()(const point& p)
  {
    assert(buffer_);
    return row(p.row())[p.col()];
  }

  template <typename V>
  inline const V& host_image2d<V>::operator()(const point& p) const
  {
    assert(buffer_);
    return row(p.row())[p.col()];
  }

  template <typename V>
  inline V* host_image2d<V>::row(unsigned i)
  {
    assert(buffer_);
    return (V*)(((char*)buffer_) + i * pitch_);
  }

  template <typename V>
  inline const V* host_image2d<V>::row(unsigned i) const
  {
    assert(buffer_);
    return (V*)(((char*)buffer_) + i * pitch_);
  }

  template <typename V>
  inline V&
  host_image2d<V>::operator()(int r, int c)
  {
    assert(buffer_);
    return operator()(point(r, c));
  }

  template <typename V>
  inline const V&
  host_image2d<V>::operator()(int r, int c) const
  {
    assert(buffer_);
    return operator()(point(r, c));
  }
}

#endif

