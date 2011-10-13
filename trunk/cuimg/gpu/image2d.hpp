#ifndef CUIMG_IMAGE2D_HPP_
# define CUIMG_IMAGE2D_HPP_

# include <cuimg/gpu/image2d.h>
# include <cuimg/error.h>
# include <cuimg/gpu/kernel_util.h>

namespace cuimg
{
  template <typename V, template <class> class PT>
  image2d<V, PT>::image2d()
  {
  }

  template <typename V, template <class> class PT>
  image2d<V, PT>::image2d(unsigned nrows, unsigned ncols)
    : domain_(nrows, ncols)
  {
    V* b;
    //cudaMallocHost((void**)&b, domain_.ncols() * sizeof(V) * domain_.nrows());
    //pitch_ = domain_.ncols() * sizeof(V);
    cudaMallocPitch((void**)&b, &pitch_, domain_.ncols() * sizeof(V), domain_.nrows());
    check_cuda_error();
    assert(b);
    //data_ = PT<V>(b, cudaFreeHost);
    data_ = PT<V>(b, cudaFree);
    data_ptr_ = data_.get();
  }

  template <typename V, template <class> class PT>
  image2d<V, PT>::image2d(V* data, unsigned nrows, unsigned ncols, unsigned pitch)
    : domain_(nrows, ncols),
      pitch_(pitch),
      data_(data),
      data_ptr_(data)
  {
  }

  template <typename V, template <class> class PT>
  image2d<V, PT>::image2d(const domain_type& d)
    : domain_(d)
  {
    V* b;
    //cudaMallocHost((void**)&b, domain_.ncols() * sizeof(V) * domain_.nrows());
    //pitch_ = domain_.ncols() * sizeof(V);
    cudaMallocPitch((void**)&b, &pitch_, domain_.ncols() * sizeof(V), domain_.nrows());
    check_cuda_error();
    assert(b);
    //data_ = PT<V>(b, cudaFreeHost);
    data_ = PT<V>(b, cudaFree);
    data_ptr_ = data_.get();
  }

  template <typename V, template <class> class PT>
  image2d<V, PT>::image2d(const image2d<V, PT>& img)
    : domain_(img.domain()),
      pitch_(img.pitch()),
      data_(img.data_sptr()),
      data_ptr_(const_cast<V*>(img.data()))
  {
  }

  template <typename V, template <class> class PT>
  image2d<V, PT>&
  image2d<V, PT>::operator=(const image2d<V, PT>& img)
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

  template <typename A, template <class> class AP, typename E>
  inline void
  assign(image2d<A, AP>& out, expr<E>& e, dim3 dimblock = dim3(16, 16))
  {
    dim3 dimgrid = grid_dimension(out.domain(), dimblock);
    internal::assign_kernel<<<dimgrid, dimblock>>>(mki(out), *(E*)&e);
   // internal::assign_kernel<<<dimgrid, dimblock>>>(mki(out));
    check_cuda_error();
  }


  template <typename A, template <class> class AP>
  inline void
  assign2(image2d<A, AP>& out, dim3 dimblock = dim3(16, 16))
  {
    dim3 dimgrid = grid_dimension(out.domain(), dimblock);
    internal::assign_kernel2<<<dimgrid, dimblock>>>(kernel_image2d<A>(out), i_float4(0.5f, 0.5f, 0.5f, 0.f));
  }


  template <typename V, template <class> class PT>
  template <typename E>
  image2d<V, PT>&
  image2d<V, PT>::operator=(expr<E>& e)
  {
    assign(*this, e);
    return *this;
  }

  template <typename V, template <class> class PT>
  const typename image2d<V, PT>::domain_type& image2d<V, PT>::domain() const
  {
    return domain_;
  }

  template <typename V, template <class> class PT>
  unsigned image2d<V, PT>::nrows() const
  {
    return domain_.nrows();
  }
  template <typename V, template <class> class PT>
  unsigned image2d<V, PT>::ncols() const
  {
    return domain_.ncols();
  }

  template <typename V, template <class> class PT>
  size_t image2d<V, PT>::pitch() const
  {
    return pitch_;
  }

  template <typename V, template <class> class PT>
  V* image2d<V, PT>::data() const
  {
    return data_ptr_;
  }
/*
  template <typename V, template <class> class PT>
  const V* image2d<V, PT>::data() const
  {
    return data_ptr_;
  }
*/
  template <typename V, template <class> class PT>
  bool image2d<V, PT>::has(const point& p) const
  {
    return domain_.has(p);
  }

  template <typename V, template <class> class PT>
  const PT<V>
  image2d<V, PT>::data_sptr() const
  {
    return data_;
  }

  template <typename V, template <class> class PT>
  PT<V>
  image2d<V, PT>::data_sptr()
  {
    return data_;
  }

  template <typename V, template <class> class PT>
  V
  image2d<V, PT>::read_back_pixel(const point& p) const
  {
    V res;
    cudaMemcpy(&res, ((char*)data_ptr_) + p.row() * pitch_ + p.col() * sizeof(V),
               sizeof(V), cudaMemcpyDeviceToHost);
    return res;
  }

}

#endif
