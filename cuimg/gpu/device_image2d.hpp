#ifndef CUIMG_IMAGE2D_HPP_
# define CUIMG_IMAGE2D_HPP_

# include <cuimg/gpu/cuda.h>
# include <cuimg/gpu/device_image2d.h>
# include <cuimg/gpu/kernel_image2d.h>
# include <cuimg/error.h>
# include <cuimg/free.h>
# include <cuimg/gpu/kernel_util.h>

namespace cuimg
{

  template <typename V>
  device_image2d<V>::device_image2d()
    : data_ptr_(0),
      begin_(0),
      border_(0)
  {
  }

  template <typename V>
  device_image2d<V>::device_image2d(unsigned nrows, unsigned ncols, unsigned _border)
    : domain_(nrows, ncols), border_(_border)
  {
    V* b = 0;

#ifndef NO_CUDA
    cudaMallocPitch((void**)&b, &pitch_, (domain_.ncols() + 2 * border_) * sizeof(V), domain_.nrows() + 2 * border_);
    check_cuda_error();
    data_ = PT(b, cudaFree);
#endif

    assert(b);

    data_ptr_ = data_.get();
    begin_ = (V*)((char*)data_ptr_ + border_ * pitch_ + border_ * sizeof(V));
  }

  template <typename V>
  device_image2d<V>::device_image2d(V* data, unsigned nrows, unsigned ncols, unsigned pitch)
    : domain_(nrows, ncols),
      pitch_(pitch),
      data_ptr_(data),
      border_(0)
  {
    data_ = PT(data, dummy_free<V>);
  }

  template <typename V>
  device_image2d<V>::device_image2d(const domain_type& d, unsigned _border)
    : domain_(d),
      border_(_border)
  {
    V* b = 0;

#ifndef NO_CUDA
    cudaMallocPitch((void**)&b, &pitch_, (domain_.ncols() + 2 * border_) * sizeof(V), domain_.nrows() + 2 * border_);
    check_cuda_error();
    data_ = PT(b, cudaFree);
#endif

    assert(b);

    data_ptr_ = data_.get();
    begin_ = (V*)((char*)data_ptr_ + border_ * pitch_ + border_ * sizeof(V));
  }

  template <typename V>
  device_image2d<V>::device_image2d(const device_image2d<V>& img)
    : domain_(img.domain()),
      pitch_(img.pitch()),
      data_(img.data_sptr()),
      data_ptr_(const_cast<V*>(img.data())),
      begin_(img.begin()),
      border_(img.border())
  {
  }

  template <typename V>
  device_image2d<V>&
  device_image2d<V>::operator=(const device_image2d<V>& img)
  {
    domain_ = img.domain();
    pitch_ = img.pitch();
    data_ = img.data_sptr();
    data_ptr_ = img.data();
    begin_ = img.begin();
    border_ = img.border();
    return *this;
  }


  template <typename V>
  void
  device_image2d<V>::swap(device_image2d<V>& o)
  {
    std::swap(domain_, o.domain_);
    std::swap(pitch_, o.pitch_);
    std::swap(data_ptr_, o.data_ptr_);
    std::swap(begin_, o.begin_);
    data_.swap(o.data_);
    std::swap(border_, o.border_);
  }

  namespace internal
  {
#ifdef NVCC
    template <typename I, typename E>
    __global__ void assign_kernel(kernel_image2d<I> out, E e)
    {
      i_int2 p = thread_pos2d();

      if (out.has(p))
        out(p) = e.eval(p);
    }

    template <typename I, typename S>
    __global__ void assign_kernel2(kernel_image2d<I> out, const S s)
    {
      //E exp = *(E*)(&e);
      i_int2 p = thread_pos2d();

      if (out.has(p))
//        out(p) = ((E*)(&e))->eval(p);
        out(p) = out(p) + s;//i_float4(0.5f, 0.5f, 0.5f, 0.f);
    }
#endif
  }

  template <typename A, typename E>
  inline void
  assign(device_image2d<A>& out, const expr<E>& e, dim3 dimblock = dim3(16, 16))
  {
    dim3 dimgrid = grid_dimension(out.domain(), dimblock);
#ifdef NVCC
    internal::assign_kernel<<<dimgrid, dimblock>>>(mki(out), *(E*)&e);
   // internal::assign_kernel<<<dimgrid, dimblock>>>(mki(out));
    check_cuda_error();
#endif
  }


  template <typename A>
  inline void
  assign2(device_image2d<A>& out, dim3 dimblock = dim3(16, 16))
  {
    dim3 dimgrid = grid_dimension(out.domain(), dimblock);
#ifdef NVCC
    internal::assign_kernel2<<<dimgrid, dimblock>>>(kernel_image2d<A>(out), i_float4(0.5f, 0.5f, 0.5f, 0.f));
#endif
  }


  template <typename V>
  template <typename E>
  device_image2d<V>&
  device_image2d<V>::operator=(const expr<E>& e)
  {
    assign(*this, e);
    return *this;
  }

  template <typename V>
  const typename device_image2d<V>::domain_type& device_image2d<V>::domain() const
  {
    return domain_;
  }

  template <typename V>
  unsigned device_image2d<V>::nrows() const
  {
    return domain_.nrows();
  }
  template <typename V>
  unsigned device_image2d<V>::ncols() const
  {
    return domain_.ncols();
  }

  template <typename V>
  size_t device_image2d<V>::pitch() const
  {
    return pitch_;
  }

  template <typename V>
  i_int2 device_image2d<V>::index_to_point(unsigned int idx) const
  {
    int r = (idx * sizeof(V)) / pitch_;
    int c = idx - r * (pitch_ / sizeof(V));
    return i_int2(r, c);
  }

  template <typename V>
  V* device_image2d<V>::begin() const
  {
    return begin_;
  }

  template <typename V>
  V* device_image2d<V>::data() const
  {
    return data_ptr_;
  }

  template <typename V>
  V* device_image2d<V>::end() const
  {
    return begin_ + (pitch_ * nrows()) / sizeof(V);
  }

  template <typename V>
  bool device_image2d<V>::has(const point& p) const
  {
    return domain_.has(p);
  }

  template <typename V>
  const typename device_image2d<V>::PT
  device_image2d<V>::data_sptr() const
  {
    return data_;
  }

  template <typename V>
  typename device_image2d<V>::PT
  device_image2d<V>::data_sptr()
  {
    return data_;
  }

  template <typename V>
  V
  device_image2d<V>::read_back_pixel(const point& p) const
  {
    V res;
# ifndef NO_CUDA
    cudaMemcpy(&res, ((char*)begin_) + p.row() * pitch_ + p.col() * sizeof(V),
               sizeof(V), cudaMemcpyDeviceToHost);
# endif
    return res;
  }

  template <typename V>
  void
  device_image2d<V>::set_pixel(const point& p, const V& v)
  {
# ifndef NO_CUDA
    cudaMemcpy(((char*)begin_) + p.row() * pitch_ + p.col() * sizeof(V), &v,
               sizeof(V), cudaMemcpyHostToDevice);
# endif
  }

  template <typename V>
  inline V* device_image2d<V>::row(int i)
  {
    assert(begin_);
    return (V*)(((char*)begin_) + i * pitch_);
  }

  template <typename V>
  inline const V* device_image2d<V>::row(int i) const
  {
    assert(begin_);
    return (V*)(((char*)begin_) + i * pitch_);
  }

}

#endif
