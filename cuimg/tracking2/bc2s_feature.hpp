#ifndef CUIMG_BC2S_FEATURE_HPP_
# define CUIMG_BC2S_FEATURE_HPP_

# include <opencv2/opencv.hpp>
# include <cuimg/profiler.h>
# include <cuimg/architectures.h>
# include <cuimg/neighb2d_data.h>
//# include <cuimg/gpu/local_jet_static.h>

#ifndef NO_CUDA
# include <cuimg/gpu/gaussian_blur.h>
#endif

namespace cuimg
{

  namespace bc2s_internals
  {
    template <typename O>
    __host__ __device__ inline int
    distance(const O& o, const bc2s& a, const i_short2& n, const unsigned scale = 1)
    {
      int d = 0;

      const typename O::V* data = &o.s1_(n);

      // return 0;
      // return o.s1_(n).x; // fixme

      if (scale == 1)
      {
	for(int i = 0; i < 8; i ++)
	{
	  int v = data[o.offsets_s1(i)].x;
	  d += ::abs(v - a[i]);
	}
      }
      //else
      {
	data = &o.s2_(n);
	for(int i = 0; i < 8; i ++)
	{
	  int v = data[o.offsets_s2(i)].x;
	  d += ::abs(v - a[8+i]);
	}
      }

      // return d / (255.f * 16.f);
      return d;
    }

    template <typename O>
    __host__ __device__ inline int
    distance(const O& o, const bc2s_256& a, const i_short2& n, const unsigned scale = 1)
    {
      int d = 0;

      const typename O::V* data = &o.s1_(n);

      if (scale == 1)
      {
	for(int i = 0; i < 16; i ++)
	{
	  int v = data[o.offsets_s1(i)].x;
	  d += ::abs(v - a[i]);
	}
      }
      //else
      {
	data = &o.s2_(n);
	for(int i = 0; i < 16; i ++)
	{
	  int v = data[o.offsets_s2(i)].x;
	  d += ::abs(v - a[16+i]);
	}
      }

      return d;
    }

    template <typename O>
    __host__ __device__ inline bc2s
    compute_feature(const O& o, const i_int2& n)
    {
      bc2s b;

      const typename O::V* data = &o.s1_(n);
      for(int i = 0; i < 8; i ++)
        b[i] = data[o.offsets_s1(i)].x;

      data = &o.s2_(n);
      for(int i = 0; i < 8; i ++)
        b[i+8] = data[o.offsets_s2(i)].x;

      return b;
    }

#ifndef NO_CPP0X
    template <typename O>
    __host__ __device__ inline bc2s_256
    compute_feature_256(const O& o, const i_int2& n)
    {
      bc2s_256 b;

      const auto* data = &o.s1_(n);
      for(int i = 0; i < 16; i ++)
        b[i] = data[o.offsets_s1_[i]].x;

      data = &o.s2_(n);
      for(int i = 0; i < 16; i ++)
        b[i+16] = data[o.offsets_s2_[i]].x;

      return b;
    }


    template <typename O>
    __host__ __device__ inline int
    distance64(const O& o, const bc2s64& a, const i_short2& n)
    {
      int d = 0;

      const auto* data = &o.s1_(n);
      for(int i = 0; i < 4; i ++)
      {
        int v = data[o.offsets_s1_[i]].x;
        d += ::abs(v - a[i]);
      }

      data = &o.s2_(n);
      for(int i = 0; i < 4; i ++)
      {
      	int v = data[o.offsets_s2_[i]].x;
      	d += ::abs(v - a[4+i]);
      }

      // return d / (255.f * 16.f);
      return d * 2;
    }

    template <typename O>
    __host__ __device__ inline bc2s64
    compute_feature64(const O& o, const i_int2& n)
    {
      bc2s64 b;

      const auto* data = &o.s1_(n);
      for(int i = 0; i < 4; i ++)
        b[i] = data[o.offsets_s1_[i]].x;

      data = &o.s2_(n);
      for(int i = 0; i < 4; i ++)
        b[i+4] = data[o.offsets_s2_[i]].x;

      return b;
    }

#endif

  }

  template <typename A>
  bc2s_feature<A>::bc2s_feature(const obox2d& d)
    : s1_(d),
      s2_(d),
      tmp_(d)
  {
    for (unsigned i = 0; i < 16; i += 2)
    {
      point2d<int> p(10,10);
      i_int2 o = circle_r3_h[i];
      i_int2 o2 = o*2;

      //std::cout << '-'<< o << std::endl;
      offsets_s1_[i/2] = (int(s1_.pitch()) * o.r()) / sizeof(V) + o.c();
      offsets_s2_[i/2] = (int(s2_.pitch()) * o2.r()) / sizeof(V) + o2.c();


      //offsets_s1_[i/2] = (long(&s1_(p + i_int2(circle_r3[i]))) - long(&s1_(p))) / sizeof(V);
      //offsets_s2_[i/2] = (long(&s2_(p + i_int2(circle_r3[i])*2)) - long(&s2_(p))) / sizeof(V);
    }

    for (unsigned i = 0; i < 8; i ++)
      std::cout << offsets_s1_[i] << "  " << offsets_s2_[i] << std::endl;
    std::cout <<  " ------------- "  << std::endl;

    // for (unsigned i = 0; i < 8; i ++)
    // {
    //   point2d<int> p(10,10);
    //   offsets_s1_[i] = (long(&s1_(p + i_int2(c8[i])*2)) - long(&s1_(p))) / sizeof(V);
    //   offsets_s2_[i] = (long(&s1_(p + i_int2(c8[i])*6)) - long(&s1_(p))) / sizeof(V);
    // }

  }

  // template <template <class> class I>
  // bc2s_feature<A>::bc2s_feature(const bc2s_feature& f)
  // {
  //   *this = f;
  // }

  // template <template <class> class I>
  // bc2s_feature<A>&
  // bc2s_feature<A>::operator=(const bc2s_feature& f)
  // {
  //   s1_ = clone(f.s1_);
  //   s2_ = clone(f.s2_);
  //   tmp_ = clone(f.tmp_);
  //   for (unsigned i = 0; i < 8; i ++)
  //   {
  //     offsets_s1_[i] = f.offsets_s1_[i];
  //     offsets_s2_[i] = f.offsets_s2_[i];
  //   }

  //   return *this;
  // }

  template <typename A>
  bc2s64_feature<A>::bc2s64_feature(const obox2d& d)
    : bc2s_feature<A>(d)
  {
    for (unsigned i = 0; i < 16; i += 4)
    {
      point2d<int> p(10,10);
      this->offsets_s1_[i/4] = (long(&this->s1_(p + i_int2(circle_r3[i]))) - long(&this->s1_(p))) / sizeof(V);
      this->offsets_s2_[i/4] = (long(&this->s2_(p + i_int2(circle_r3[i])*2)) - long(&this->s2_(p))) / sizeof(V);
    }
  }


  template <typename A>
  void
  bc2s_feature<A>::update(const I& in, const cpu&)
  {
    SCOPE_PROF(bc2s_feature::update);
    dim3 dimblock = ::cuimg::dimblock(cpu(), sizeof(V) + sizeof(i_uchar1), in.domain());

    //local_jet_static_<0, 0, 2, 2>::run(in, s1_, tmp_, 0, dimblock);
    // local_jet_static_<0, 0, 3, 3>::run(in, s2_, tmp_, 0, dimblock);
    //local_jet_static_<0, 0, 1, 1>::run(s1_, s2_, tmp_, 0, dimblock);

    cv::GaussianBlur(cv::Mat(in), cv::Mat(s1_), cv::Size(7, 7), 2, 2, cv::BORDER_REPLICATE);
    cv::GaussianBlur(cv::Mat(s1_), cv::Mat(s2_), cv::Size(3, 3), 1, 1, cv::BORDER_REPLICATE);
    //cv::GaussianBlur(cv::Mat(in), cv::Mat(s2_), cv::Size(9, 9), 3, 3, cv::BORDER_REPLICATE);
    // dg::widgets::ImageView("frame") << (*(host_image2d<i_uchar1>*)(&s2_)) << dg::widgets::show;
    // dg::pause();
  }

#ifndef NO_CUDA

  template <typename A>
  void
  bc2s_feature<A>::update(const I& in, const cuda_gpu&)
  {
    SCOPE_PROF(bc2s_feature::update);
    //local_jet_static_<0, 0, 2, 2>::run(in, s1_, tmp_, 0, dimblock);
    //local_jet_static_<0, 0, 3, 3>::run(in, s2_, tmp_, 0, dimblock);
    //local_jet_static_<0, 0, 1, 1>::run(s1_, s2_, tmp_, 0, dimblock);
    gaussian_blur(in, s1_, tmp_, 2, 3);
    gaussian_blur(s1_, s2_, tmp_, 1, 1);
    //cv::GaussianBlur(cv::Mat(in), cv::Mat(s2_), cv::Size(9, 9), 3, 3, cv::BORDER_REPLICATE);
    // dg::widgets::ImageView("frame") << (*(host_image2d<i_uchar1>*)(&s2_)) << dg::widgets::show;
    // dg::pause();
  }
#endif

  template <typename A>
  int
  bc2s_feature<A>::distance(const bc2s& a, const i_short2& n, unsigned scale)
  {
    return bc2s_internals::distance(*this, a, n, scale);
  }

  template <typename A>
  bc2s
  bc2s_feature<A>::operator()(const i_short2& p) const
  {
    return bc2s_internals::compute_feature(*this, p);
  }

#ifndef NVCC

  template <typename A>
  int
  bc2s64_feature<A>::distance(const bc2s64& a, const i_short2& n)
  {
    return bc2s_internals::distance64(*this, a, n);
  }

  template <typename A>
  bc2s64
  bc2s64_feature<A>::operator()(const i_short2& p) const
  {
    return bc2s_internals::compute_feature64(*this, p);
  }

#endif

  template <typename A>
  int
  bc2s_feature<A>::offsets_s1(int o) const
  {
    return offsets_s1_[o];
  }

  template <typename A>
  int
  bc2s_feature<A>::offsets_s2(int o) const
  {
    return offsets_s2_[o];
  }


#ifndef NO_CUDA
  /// cuda_bc2s_feature
  /// ##################

  cuda_bc2s_feature::cuda_bc2s_feature(bc2s_feature<cuda_gpu>& o)
    : s1_(o.s1_),
      s2_(o.s2_)
  {
    // s1_ = o.s1_;
    // s2_ = o.s2_;

    //if (not cuda_bc2s_offsets_loaded_)
    {
      cuda_bc2s_offsets_loaded_ = true;
      std::cout << sizeof(o.offsets_s1_) << std::endl;
      cudaMemcpyToSymbol(cuda_bc2s_offsets_s1, o.offsets_s1_, sizeof(o.offsets_s1_));
      cudaMemcpyToSymbol(cuda_bc2s_offsets_s2, o.offsets_s2_, sizeof(o.offsets_s2_));
    }
  }

  __device__ int
  cuda_bc2s_feature::distance(const bc2s& a, const i_short2& n, unsigned scale)
  {
    return bc2s_internals::distance(*this, a, n, scale);
  }

  __device__ bc2s
  cuda_bc2s_feature::operator()(const i_short2& p) const
  {
    return bc2s_internals::compute_feature(*this, p);
  }

  __device__ int
  cuda_bc2s_feature::offsets_s1(int o) const
  {
    return cuda_bc2s_offsets_s1[o];
  }

  __device__ int
  cuda_bc2s_feature::offsets_s2(int o) const
  {
    return cuda_bc2s_offsets_s2[o];
  }

#endif

  /// bc2s_256
  /// ##################

  template <template <class> class I>
  bc2s_256_feature<I>::bc2s_256_feature(const obox2d& d)
    : s1_(d),
      s2_(d),
      tmp_(d)
  {
    for (unsigned i = 0; i < 16; i ++)
    {
      point2d<int> p(10,10);
      offsets_s1_[i] = (long(&s1_(p + i_int2(circle_r3[i]))) - long(&s1_(p))) / sizeof(V);
      offsets_s2_[i] = (long(&s2_(p + i_int2(circle_r3[i])*2)) - long(&s2_(p))) / sizeof(V);
    }
  }

  template <template <class> class I>
  void
  bc2s_256_feature<I>::update(const I<gl8u>& in)
  {
    dim3 dimblock = ::cuimg::dimblock(cpu(), sizeof(V) + sizeof(i_uchar1), in.domain());

    // local_jet_static_<0, 0, 2, 2>::run(in, s1_, tmp_, 0, dimblock);
    // local_jet_static_<0, 0, 3, 3>::run(in, s2_, tmp_, 0, dimblock);
    //local_jet_static_<0, 0, 1, 1>::run(s1_, s2_, tmp_, 0, dimblock);
  }

  template <template <class> class I>
  int
  bc2s_256_feature<I>::distance(const bc2s_256& a, const i_short2& n, unsigned scale)
  {
    return bc2s_internals::distance(*this, a, n, scale);
  }

  template <template <class> class I>
  bc2s_256
  bc2s_256_feature<I>::operator()(const i_short2& p) const
  {
    //return bc2s_internals::compute_feature_256(*this, p);
  }

  // ########## bc2s64 feature vector methods. ###########
  // ###################################################

  bc2s::bc2s()
  {}

  bc2s::bc2s(const float4& o)
  {
    tex_float = o;
  }

  bc2s::bc2s(const bc2s& o)
  {
    tex_float = o.tex_float;
  }

  bc2s&
  bc2s::operator=(const bc2s& o)
  {
    tex_float = o.tex_float;
    return *this;
  }


  const unsigned char&
  bc2s::operator[](const unsigned i) const
  {
    return distances[i];
  }


  unsigned char&
  bc2s::operator[](const unsigned i)
  {
    return distances[i];
  }




  // ########## bc2s_256 feature vector methods. ###########
  // ###################################################

  bc2s_256::bc2s_256()
  {}

  // bc2s_256::bc2s_256(const float2& o)
  // {
  //   tex_float = o;
  // }

  bc2s_256::bc2s_256(const bc2s_256& o)
  {
    tex_float[0] = o.tex_float[0];
    tex_float[1] = o.tex_float[1];
  }

  bc2s_256&
  bc2s_256::operator=(const bc2s_256& o)
  {
    tex_float[0] = o.tex_float[0];
    tex_float[1] = o.tex_float[1];
    return *this;
  }


  const unsigned char&
  bc2s_256::operator[](const unsigned i) const
  {
    return distances[i];
  }

  unsigned char&
  bc2s_256::operator[](const unsigned i)
  {
    return distances[i];
  }


  // ########## bc2s64 feature vector methods. ###########
  // ###################################################

  bc2s64::bc2s64()
  {}

  // bc2s64::bc2s64(const float2& o)
  // {
  //   tex_float = o;
  // }

  bc2s64::bc2s64(const bc2s64& o)
  {
    tex_float = o.tex_float;
  }

  bc2s64&
  bc2s64::operator=(const bc2s64& o)
  {
    tex_float = o.tex_float;
    return *this;
  }


  const unsigned char&
  bc2s64::operator[](const unsigned i) const
  {
    return distances[i];
  }


  unsigned char&
  bc2s64::operator[](const unsigned i)
  {
    return distances[i];
  }


}

#endif
