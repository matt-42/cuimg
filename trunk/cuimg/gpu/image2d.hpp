#ifndef CUIMG_IMAGE2D_HPP_
# define CUIMG_IMAGE2D_HPP_

# include <cuda.h>
# include <cuda_runtime.h>
# include <cuimg/gpu/image2d.h>
# include <cuimg/gpu/kernel_image2d.h>
# include <cuimg/error.h>
# include <cuimg/gpu/kernel_util.h>

namespace cuimg
{


  template <typename T>
  void dummy_free(T* p)
  {
  }

  template <typename V>
  image2d<V>::image2d()
  {
  }

  template <typename V>
  image2d<V>::image2d(unsigned nrows, unsigned ncols)
    : domain_(nrows, ncols)
  {
    V* b;

    cudaMallocPitch((void**)&b, &pitch_, domain_.ncols() * sizeof(V), domain_.nrows());
    check_cuda_error();
    data_ = PT(b, cudaFree);

    assert(b);

    data_ptr_ = data_.get();
  }

  template <typename V>
  image2d<V>::image2d(V* data, unsigned nrows, unsigned ncols, unsigned pitch)
    : domain_(nrows, ncols),
      pitch_(pitch),
      data_ptr_(data)
  {
    data_ = PT(data, dummy_free<V>);
  }

  template <typename V>
  image2d<V>::image2d(const domain_type& d)
    : domain_(d)
  {
    V* b;

    cudaMallocPitch((void**)&b, &pitch_, domain_.ncols() * sizeof(V), domain_.nrows());
    check_cuda_error();
    assert(b);
    data_ = PT(b, cudaFree);

    data_ptr_ = data_.get();
  }

  template <typename V>
  image2d<V>::image2d(const image2d<V>& img)
    : domain_(img.domain()),
      pitch_(img.pitch()),
      data_(img.data_sptr()),
      data_ptr_(const_cast<V*>(img.data()))
  {
  }

  template <typename V>
  image2d<V>&
  image2d<V>::operator=(const image2d<V>& img)
  {
    domain_ = img.domain();
    pitch_ = img.pitch();
    data_ = img.data_sptr();
    data_ptr_ = img.data();
    return *this;
  }

  namespace internal
  {
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
  }

  template <typename A, typename E>
  inline void
  assign(image2d<A>& out, const expr<E>& e, dim3 dimblock = dim3(16, 16))
  {
    dim3 dimgrid = grid_dimension(out.domain(), dimblock);
#ifdef NVCC
    internal::assign_kernel<<<dimgrid, dimblock>>>(mki(out), *(E*)&e);
#endif
   // internal::assign_kernel<<<dimgrid, dimblock>>>(mki(out));
    check_cuda_error();
  }


  template <typename A>
  inline void
  assign2(image2d<A>& out, dim3 dimblock = dim3(16, 16))
  {
    dim3 dimgrid = grid_dimension(out.domain(), dimblock);
#ifdef NVCC
    internal::assign_kernel2<<<dimgrid, dimblock>>>(kernel_image2d<A>(out), i_float4(0.5f, 0.5f, 0.5f, 0.f));
#endif
  }


  template <typename V>
  template <typename E>
  image2d<V>&
  image2d<V>::operator=(const expr<E>& e)
  {
    assign(*this, e);
    return *this;
  }

  template <typename V>
  const typename image2d<V>::domain_type& image2d<V>::domain() const
  {
    return domain_;
  }

  template <typename V>
  unsigned image2d<V>::nrows() const
  {
    return domain_.nrows();
  }
  template <typename V>
  unsigned image2d<V>::ncols() const
  {
    return domain_.ncols();
  }

  template <typename V>
  size_t image2d<V>::pitch() const
  {
    return pitch_;
  }

  template <typename V>
  V* image2d<V>::data() const
  {
    return data_ptr_;
  }

  template <typename V>
  bool image2d<V>::has(const point& p) const
  {
    return domain_.has(p);
  }

  template <typename V>
  const typename image2d<V>::PT
  image2d<V>::data_sptr() const
  {
    return data_;
  }

  template <typename V>
  typename image2d<V>::PT
  image2d<V>::data_sptr()
  {
    return data_;
  }

  template <typename V>
  V
  image2d<V>::read_back_pixel(const point& p) const
  {
    V res;
    cudaMemcpy(&res, ((char*)data_ptr_) + p.row() * pitch_ + p.col() * sizeof(V),
               sizeof(V), cudaMemcpyDeviceToHost);
    return res;
  }

  template <typename V>
  void
  image2d<V>::set_pixel(const point& p, const V& v)
  {
    cudaMemcpy(((char*)data_ptr_) + p.row() * pitch_ + p.col() * sizeof(V), &v,
               sizeof(V), cudaMemcpyHostToDevice);
  }

}

#endif
