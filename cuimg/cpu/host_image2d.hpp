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
    : begin_(0)
  {
  }


  template <typename V>
  host_image2d<V>::~host_image2d()
  {
    data_.reset();
  }

  template <typename V>
  host_image2d<V>::host_image2d(unsigned nrows, unsigned ncols, unsigned border, bool pinned)
    : domain_(nrows, ncols),
      border_(border)
  {
    allocate(domain_, border, pinned);
  }

  template <typename V>
  host_image2d<V>::host_image2d(V* data, unsigned nrows, unsigned ncols, unsigned pitch)
    : domain_(nrows, ncols),
      pitch_(pitch),
      border_(0)
  {
    data_ = PT(data, dummy_free<V>);
    begin_ = data_.get();
  }

  template <typename V>
  host_image2d<V>::host_image2d(const domain_type& d, unsigned border, bool pinned)
    : domain_(d),
      pitch_(0),
      border_(border),
      data_(),
      begin_(0)
  {
    allocate(d, border, pinned);
  }

  template <typename V>
  struct free_array_of_object
  {
    free_array_of_object(host_image2d<V>& i)
      : img(i)
    {}

    void operator()(V* ptr)
    {
      for (unsigned r = 0; r < img.nrows(); r++)
	for (unsigned c = 0; c < img.ncols(); c++)
	  img(r, c).~V();
      delete [] (char*)ptr;
    }

  private:
    host_image2d<V>& img;
  };

  template <typename V>
  void
  host_image2d<V>::allocate(const domain_type& d, unsigned border, bool pinned)
  {
    V* ptr = 0;

#ifndef NO_CUDA
    if (pinned)
    {
      //cudaMallocHost(&ptr, domain_.nrows() * domain_.ncols() * sizeof(V));
      cudaMallocHost(&ptr, (domain_.nrows() + 2 * border) * (domain_.ncols() + 2 * border) * sizeof(V));
      pitch_ = domain_.ncols() * sizeof(V);
      data_ = boost::shared_ptr<V>(ptr, cudaFreeHost);
    }
    else
#endif
    {
      pitch_ = 0;
      pitch_ = (d.ncols() + 2 * border) * sizeof(V);
      if (pitch_ % 4)
       	pitch_ = pitch_ + 4 - (pitch_ & 3);
      ptr = (V*) new char[(d.nrows() + 2 * border) * pitch_ + 64];
      data_ = boost::shared_ptr<V>(ptr, array_free<V>);
      // data_ = boost::shared_ptr<V>(ptr, [&] (V* ptr) {
      // 	  for (unsigned r = 0; r < this->nrows(); r++)
      // 	    for (unsigned c = 0; c < this->ncols(); c++)
      // 	      this->operator()(r, c).~V();
      // 	  delete [] (char*)ptr;
      // 	});
      //data_ = boost::shared_ptr<V>(ptr, free_array_of_object<V>(*this));
      // assert(!(size_t(begin_) % 64));
      // assert(!(size_t(pitch_) % 64));
    }

    begin_ = data_.get() + (border * pitch_) / sizeof(V) + border;
    // if (size_t(begin_) % 64)
    //   begin_ = begin_ + 64 - (size_t(begin_) % 64);

    // for (unsigned r = 0; r < nrows(); r++)
    //   for (unsigned c = 0; c < ncols(); c++)
    // 	new(&this->operator()(r, c)) V();

    assert(begin_);
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

//     begin_ = data_.get();

//     begin_ = data_.get();
//   }



#ifdef WITH_OPENCV
	namespace internal
	{
		template <typename V>
		struct opencv_release_ref
		{
			opencv_release_ref(cv::Mat _m)
				: m(_m)
			{
			}
			void operator()(V* v)
			{
			}
			cv::Mat m;
		};
	}
  template <typename V>
  host_image2d<V>::host_image2d(IplImage* imgIpl)
  {
    *this = imgIpl;
  }

  template <typename V>
  host_image2d<V>::host_image2d(cv::Mat m)
  {
    assert(m.rows > 0 && m.cols > 0);
		internal::opencv_release_ref<V> rel(m);
    pitch_ = m.step;
    data_ = PT((V*) m.data, rel);
    begin_ = (V*) m.data;
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
    border_ = img.border_;
    begin_ = img.begin_;
    return *this;
  }


  template <typename V>
  void
  host_image2d<V>::swap(host_image2d<V>& img)
  {
    std::swap(domain_, img.domain_);
    std::swap(pitch_, img.pitch_);
    std::swap(border_, img.border_);
    std::swap(begin_, img.begin_);
    data_.swap(img.data_);
  }

  template <typename V>
  host_image2d<V>::host_image2d(const host_image2d<V>& img)
    : domain_(img.domain()),
      pitch_(img.pitch()),
      border_(img.border_),
      data_(img.data_)
  {
    begin_ = img.begin_;
  }

#ifdef WITH_OPENCV
  template <typename V>
  host_image2d<V>&
  host_image2d<V>::operator=(IplImage *imgIpl)
  {
    pitch_ = imgIpl->widthStep;
    data_ = PT((V*) imgIpl->imageData, dummy_free<V>);
    begin_ = (V*) imgIpl->imageData;
    domain_ = domain_type(imgIpl->height,imgIpl->width);
    border_ = 0;
    return *this;
  }


  template <typename V>
  host_image2d<V>&
  host_image2d<V>::operator=(cv::Mat m)
  {
		internal::opencv_release_ref<V> rel(m);

    assert(m.rows && m.cols);

    pitch_ = m.step;
    border_ = 0;
    data_ = PT((V*) m.data, rel);
    begin_ = (V*) m.data;
    domain_ = domain_type(m.rows, m.cols);
    return *this;
  }

#endif

  template <typename V>
  const typename host_image2d<V>::domain_type& host_image2d<V>::domain() const
  {
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


  template <typename V>
  inline int host_image2d<V>::border() const
  {
    return border_;
  }


#ifdef WITH_OPENCV
  //  Get IplImage from host_image2d --
  template <typename V>
  IplImage* host_image2d<V>::getIplImage() const
  {
    assert(begin_);
    //allocate the structure
    IplImage* frameIPL = cvCreateImageHeader(cvSize(ncols(),nrows()),
					     sizeof(typename V::vtype)*8,
					     V::size);
    //init the data structure
    cvSetData(frameIPL, (void*)begin(), pitch());
    return frameIPL;
  }

  template <typename V>
  host_image2d<V>::operator cv::Mat() const
  {
    assert(begin_);
    return cv::Mat(nrows(), ncols(), opencv_typeof<V>::ret, (void*)begin(), pitch());
  }

#endif
  template <typename V>
  inline V* host_image2d<V>::data()
  {
    return data_.get();
  }

  template <typename V>
  inline const V* host_image2d<V>::data() const
  {
    return data_.get();
  }

  template <typename V>
  V* host_image2d<V>::begin() const
  {
    return begin_;
  }

  template <typename V>
  V* host_image2d<V>::end() const
  {
    return begin_ + (pitch_ * nrows()) / sizeof(V);
  }

  template <typename V>
  inline size_t host_image2d<V>::buffer_size() const
  {
    assert(begin_);
    return (nrows() + 2 * border_) * pitch_;
  }

  template <typename V>
  inline V& host_image2d<V>::operator()(const point& p)
  {
    assert(begin_);
    return row(p.row())[p.col()];
  }

  template <typename V>
  inline const V& host_image2d<V>::operator()(const point& p) const
  {
    assert(begin_);
    return row(p.row())[p.col()];
  }

  template <typename V>
  inline V* host_image2d<V>::row(int i)
  {
    assert(begin_);
    return (V*)(((char*)begin_) + i * pitch_);
  }

  template <typename V>
  inline const V* host_image2d<V>::row(int i) const
  {
    assert(begin_);
    return (V*)(((char*)begin_) + i * pitch_);
  }

  template <typename V>
  inline V&
  host_image2d<V>::operator()(int r, int c)
  {
    assert(begin_);
    return operator()(point(r, c));
  }

  template <typename V>
  inline const V&
  host_image2d<V>::operator()(int r, int c) const
  {
    assert(begin_);
    return operator()(point(r, c));
  }


  template <typename V>
  inline V&
  host_image2d<V>::operator[](int i)
  {
    assert(begin_);
    return begin_[i];
  }

  template <typename V>
  inline const V&
  host_image2d<V>::operator[](int i) const
  {
    assert(begin_);
    return begin_[i];
  }

  template <typename V>
  int host_image2d<V>::point_to_index(const point& p) const
  {
    assert(!(pitch_ % sizeof(V)));
    return (p.r() * pitch_) / sizeof(V) + p.c();
  }

  template <typename V>
  i_int2 host_image2d<V>::index_to_point(int idx) const
  {
    assert(!(pitch_ % sizeof(V)));
    int r = (idx * int(sizeof(V))) / pitch_;
    int c = idx - r * (pitch_ / int(sizeof(V)));
    return i_int2(r, c);
  }

}

#endif
