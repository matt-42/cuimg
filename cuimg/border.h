#ifndef CUIMG_BORDER_H_
# define CUIMG_BORDER_H_

# include <cuimg/profiler.h>
# include <cuimg/gpu/cuda.h>
# include <cuimg/point2d.h>
# include <cuimg/box2d.h>
# include <cuimg/run_kernel.h>

namespace cuimg
{
  class border
  {
  public:

    __host__ __device__ inline border();
    __host__ __device__ inline border(int thickness)
      : thickness_(thickness)
    {
    }

    __host__ __device__ inline int thickness() const { return thickness_; }

  private:
    int thickness_;
  };


  inline __host__ __device__
  box2d operator-(const obox2d& d, const border& bd)
  {
    int b = bd.thickness();
    return box2d(i_int2(b, b),
                 i_int2(d.nrows() - b - 1, d.ncols() - b - 1));
  }


  inline __host__ __device__
  box2d operator-(const box2d& d, const border& bd)
  {
    int b = bd.thickness();
    return box2d(i_int2(d.p1().r() + b, d.p1().c() + b),
                 i_int2(d.p2().r() - b, d.p2().c() - b));
  }


  inline __host__ __device__
  box2d operator+(const obox2d& d, const border& bd)
  {
    return d - border(-bd.thickness());
  }


  inline __host__ __device__
  box2d operator+(const box2d& d, const border& bd)
  {
    return d - border(-bd.thickness());
  }


  template <typename I>
  class no_border
  {
  public:
    typedef typename I::value_type V;
    typedef typename I::point P;

    no_border(const I& img, int offset)
      : image_(img),
	data_(img.begin() + offset)
    {
    }

    no_border(const I& img, i_int2 offset)
      : image_(img),
	data_(img.begin() + img.point_to_index(offset))
    {
    }

    inline V& operator[](int i)              { return data_[i]; }
    inline const V& operator[](int i) const  { return data_[i]; }

  private:
    const I& image_;
    V* data_;
  };

  template <typename I>
  I clone_with_border(const I& img, int border)
  {
    I tmp(img.domain(), border);
    copy(img, tmp);
    return tmp;
  }


  __host__ __device__ inline
  i_int2 clamp_coords(i_int2 p, const obox2d& domain)
  {
    if (p.r() < 0) p.r() = 0;
    if (p.r() >= domain.nrows()) p.r() = domain.nrows() - 1;

    if (p.c() < 0) p.c() = 0;
    if (p.c() >= domain.ncols()) p.c() = domain.ncols() - 1;

    return p;
  }

  template<typename I>
  struct make_border_clamp_kernel
  {
    typename I::kernel_type in_;
    box2d domain_with_border_;
    make_border_clamp_kernel(I in, box2d d)
      : in_(in),
	domain_with_border_(d)
    {
    }

    inline __host__ __device__
    void operator()(i_int2 p)
    {
      if (not in_.has(p) and domain_with_border_.has(p))
	in_(p) = in_(clamp_coords(p, in_.domain()));
    }
  };

  template <typename I>
  void fill_border_clamp(I& img)
  {
    SCOPE_PROF(fill_border_clamp);
    box2d d = img.domain() + border(img.border());
    run_kernel2d_functor(make_border_clamp_kernel<I>(img, d),
    			 d,
    			 typename I::architecture());
    check_cuda_error();
  }


  // template <typename I>
  // class border_clamp
  // {
  // public:
  //   typedef typename I::value_type V;
  //   typedef typename I::point P;

  //   border_clamp(const I& img, int offset)
  //     : image_(img),
  // 	data_(img.begin() + offset),
  // 	end_(img.end())
  //   {
  //   }

  //   border_clamp(const I& img, i_int2 offset)
  //     : image_(img),
  // 	data_(img.begin() + img.point_to_index(offset)),
  // 	end_(img.end())
  //   {
  //   }

  //   inline V& operator[](int i)              { return data_[clamp_index(i)]; }
  //   inline const V& operator[](int i) const  { return data_[clamp_index(i)]; }

  // private:
  //   inline int clamp_index(int i_)
  //   {
  //     V* i = data_ + i_;
  //     if (i >= image_.begin() && i < end_) return i_;

  //     P p = image_.index_to_point(i - image_.begin());
  //     if (p.r() < 0) p.r() = 0;
  //     if (p.r() >= image_.nrows()) p.r() = image_.nrows() - 1;

  //     if (p.c() < 0) p.c() = 0;
  //     if (p.c() >= image_.ncols()) p.c() = image_.ncols() - 1;

  //     return image_.point_to_index(p) - (data_ - image_.begin());
  //   }

  //   const I& image_;
  //   V* data_;
  //   V* end_;
  // };

}

#endif
