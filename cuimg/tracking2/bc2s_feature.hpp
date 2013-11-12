#ifndef CUIMG_BC2S_FEATURE_HPP_
# define CUIMG_BC2S_FEATURE_HPP_

# include <cuimg/profiler.h>
# include <cuimg/architectures.h>
# include <cuimg/neighb2d_data.h>
# include <cuimg/border.h>
# include <cuimg/box2d.h>

# include <cuimg/cpu/gaussian_blur.h>

# ifdef WITH_OPENCV
#  include <opencv2/opencv.hpp>
# endif

#ifndef NO_CUDA

# include <cuimg/gpu/gaussian_blur.h>
# include <cuimg/gpu/texture.h>

#endif

namespace cuimg
{

#ifndef NO_CUDA
  ::texture<uchar1, 2, cudaReadModeElementType> bc2s_tex_s1;
  ::texture<uchar1, 2, cudaReadModeElementType> bc2s_tex_s2;
#endif

  namespace bc2s_internals
  {
    template <typename O>
    __host__ __device__ inline int
    distance(const O& o, const bc2s& a, const i_short2& n, const unsigned scale = 1)
    {
      int d = 0;
      int d2 = 0;
      int wsum = 0;
      const typename O::V* data = &o.s1_(n);
      // int idx = o.s1_.point_to_index(n);

      if (scale == 1)
      {
	// auto data = B(o.s1_, n);
	for(int i = 0; i < 8; i ++)
	{
	  int v = data[o.offsets_s1(i)].x;
	  //int v = o.s1_[idx + o.offsets_s1(i)].x;
	  //d += (v - a[i]) * (v - a[i]);

	  //d += ::abs(v - a[i]);
	  d += ::abs(v - a[i]) * a.weights[i];
	  wsum += a.weights[i];
	}
	//return sqrt(d) * 6;
      }
      //else
      {
	{
	  //idx = o.s2_.point_to_index(n);
	  //auto data = B(o.s2_, n);
	  data = &o.s2_(n);
	  for(int i = 0; i < 8; i ++)
	  {
	    int v = data[o.offsets_s2(i)].x;
	    //int v = o.s2_[idx + o.offsets_s2(i)].x;

	    //d2 += ::abs(v - a[8+i]);

	    d2 += ::abs(v - a[8+i]) * a.weights[8+i];
	    wsum += a.weights[8+i];

	    //d2 += (v - a[8+i]) * (v - a[8+i]);
	  }
	  //return sqrt(d2) * 6;
	}
      }

      // return d / (255.f * 16.f);
      //return d + d2 * 5;
      //return d + 25 * d2 + 2 * 5 * d * d2;
      //return ::sqrt(d) + ::sqrt(d2) * 6;
      // if (scale == 1)
      // 	return (d + d2) / 8;
      // else
      // 	return (d + d2) / (8);

      return (d + d2) / 255;
      //return sqrt(d) * 6;
    }

#ifndef NO_CUDA
    __device__ inline int
    distance_tex(const bc2s& a, const i_short2& n, const unsigned scale = 1)
    {
      int d = 0;
      if (scale == 1)
      {
	for(int i = 0; i < 8; i ++)
	{
	  int v = tex2D(bc2s_tex_s1, n.c() + circle_r3[i][1], n.r() + circle_r3[i][0]).x;
	  d += ::abs(v - a[i]);
	}
      }
      //else
      {
	for(int i = 0; i < 8; i ++)
	{
	  int v = tex2D(bc2s_tex_s2, n.c() + circle_r3[i][1] * 2, n.r() + circle_r3[i][0] * 2).x;
	  d += ::abs(v - a[8+i]) * 5;
	}
      }

      // return d / (255.f * 16.f);
      return d;
    }
#endif

    template <typename O>
    __host__ __device__ inline bc2s
    compute_feature(const O& o, const i_int2& n)
    {
      bc2s b;
      assert((o.domain()).has(n));

      const typename O::V* data = &o.s1_(n);
      for(int i = 0; i < 8; i ++)
      {
	// assert(o.s1_.begin() <= data + o.offsets_s1(i));
	// assert(o.s1_.end() > data + o.offsets_s1(i));
        b[i] = data[o.offsets_s1(i)].x;
      }

      data = &o.s2_(n);
      for(int i = 0; i < 8; i ++)
      {
	// assert(o.s2_.begin() <= data + o.offsets_s2(i));
	// assert(o.s2_.end() > data + o.offsets_s2(i));
        b[i+8] = data[o.offsets_s2(i)].x;
      }

      return b;
    }

#ifndef NO_CUDA
    __device__ inline bc2s
    compute_feature_tex(const i_int2& n)
    {
      bc2s b;

      for(int i = 0; i < 8; i ++)
      {
	b[i] = tex2D(bc2s_tex_s1, n.c() + circle_r3[i][1], n.r() + circle_r3[i][0]).x;
      }

      for(int i = 0; i < 8; i ++)
      {
        b[i+8] = tex2D(bc2s_tex_s2, n.c() + circle_r3[i][1] * 2, n.r() + circle_r3[i][0] * 2).x;
      }

      return b;
    }

#endif

  }

  template <typename A>
  bc2s_feature<A>::bc2s_feature(const obox2d& d)
    : s1_(d, 3),
      s2_(d, 6),
      tmp_(d),
      kernel_1_(1, 1),
      kernel_2_(2, 3)
  {
    for (unsigned i = 0; i < 16; i += 2)
    {
      i_int2 o = circle_r3_h[i];
      i_int2 o2 = o*2;
      // i_int2 o = c8_h[i/2];
      // i_int2 o2 = i_int2(circle_r3_h[i]) * 2;

      offsets_s1_[i/2] = (int(s1_.pitch()) * o.r()) / sizeof(V) + o.c();
      offsets_s2_[i/2] = (int(s2_.pitch()) * o2.r()) / sizeof(V) + o2.c();
    }

    // for (unsigned i = 0; i < 8; i ++)
    // {
    //   point2d<int> p(10,10);
    //   i_int2 o = c8_h[i];
    //   i_int2 o2 = o*2;

    //   offsets_s1_[i/2] = (int(s1_.pitch()) * o.r()) / sizeof(V) + o.c();
    //   offsets_s2_[i/2] = (int(s2_.pitch()) * o2.r()) / sizeof(V) + o2.c();
    // }
  }

  // template <template <class> class I>
  // bc2s_feature<A>::bc2s_feature(const bc2s_feature& f)
  // {
  //   *this = f;
  // }


  template <typename A>
  void
  bc2s_feature<A>::swap(bc2s_feature<A>& o)
  {
    s1_.swap(o.s1_);
    s2_.swap(o.s2_);
  }

  template <typename A>
  bc2s_feature<A>&
  bc2s_feature<A>::operator=(const bc2s_feature& f)
  {
    s1_ = f.s1_;
    s2_ = f.s2_;
    tmp_ = f.tmp_;
    for (unsigned i = 0; i < 8; i ++)
    {
      offsets_s1_[i] = f.offsets_s1_[i];
      offsets_s2_[i] = f.offsets_s2_[i];
    }

    kernel_1_ = f.kernel_1_;
    kernel_2_ = f.kernel_2_;
    return *this;
  }

  template <typename A>
  void
  bc2s_feature<A>::update(const I& in, const cpu&)
  {
    SCOPE_PROF(bc2s_feature::update);

    #ifdef WITH_OPENCV
    cv::Mat opencv_s1(s1_);
    cv::Mat opencv_s2(s2_);
    cv::GaussianBlur(cv::Mat(in), opencv_s1, cv::Size(3, 3), 1, 1, cv::BORDER_REPLICATE);
    fill_border_clamp(s1_);
    cv::GaussianBlur(cv::Mat(s1_), opencv_s2, cv::Size(5, 5), 1.8, 1.8, cv::BORDER_REPLICATE);
    fill_border_clamp(s2_);

    #else

    gaussian_blur_sigma1(in, s1_, tmp_);
    gaussian_blur_sigma1(s1_, s2_, tmp_);
    fill_border_clamp(s1_);
    fill_border_clamp(s2_);
    #endif
  }

#ifndef NO_CUDA

  template <typename A>
  void
  bc2s_feature<A>::update(const I& in, const cuda_gpu&)
  {
    SCOPE_PROF(bc2s_feature::update);
    gaussian_blur(in, s1_, tmp_, kernel_2_);
    //copy(in, s1_);
    gaussian_blur(s1_, s2_, tmp_, kernel_1_);
    fill_border_clamp(s1_);
    fill_border_clamp(s2_);
  }

  template <typename A>
  void
  bc2s_feature<A>::bind(const cuda_gpu&)
  {
    bindTexture2d(s1(), bc2s_tex_s1);
    bindTexture2d(s2(), bc2s_tex_s2);
    check_cuda_error();
  }

#endif

  template <typename A>
  void
  bc2s_feature<A>::bind(const cpu&)
  {
  }

  template <typename A>
  void
  bc2s_feature<A>::bind()
  {
    bind(A());
  }

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
    return;
    // if (scales.find(o.domain().size()) == scales.end())
    // {
    //   unsigned i = scales.size();
    //   std::cout << "load- " << i << std::endl;
    //   scales[o.domain().size()] = i;
    //   switch (scales[o.domain().size()])
    //   {
    // 	case 0:
    // 	  std::cout << "load 0" << std::endl;
    // 	  cudaMemcpyToSymbol(cuda_bc2s_offsets_s1_0, o.offsets_s1_, sizeof(o.offsets_s1_));
    // 	  cudaMemcpyToSymbol(cuda_bc2s_offsets_s2_0, o.offsets_s2_, sizeof(o.offsets_s2_));
    // 	  break;
    // 	case 1:
    // 	  std::cout << "load 1" << std::endl;
    // 	  cudaMemcpyToSymbol(cuda_bc2s_offsets_s1_1, o.offsets_s1_, sizeof(o.offsets_s1_));
    // 	  cudaMemcpyToSymbol(cuda_bc2s_offsets_s2_1, o.offsets_s2_, sizeof(o.offsets_s2_));
    // 	  break;
    // 	case 2:
    // 	  std::cout << "load 2" << std::endl;
    // 	  cudaMemcpyToSymbol(cuda_bc2s_offsets_s1_2, o.offsets_s1_, sizeof(o.offsets_s1_));
    // 	  cudaMemcpyToSymbol(cuda_bc2s_offsets_s2_2, o.offsets_s2_, sizeof(o.offsets_s2_));
    // 	  break;
    //   }
    // }
    // scaleid_ = scales[o.domain().size()];

    // //if (not cuda_bc2s_offsets_loaded_)
    // {
    //   //cuda_bc2s_offsets_loaded_ = true;
    //   //cudaMemcpyToSymbol(cuda_bc2s_offsets_s1, o.offsets_s1_, sizeof(o.offsets_s1_));
    //   //cudaMemcpyToSymbol(cuda_bc2s_offsets_s2, o.offsets_s2_, sizeof(o.offsets_s2_));

    //   cudaMemcpyToSymbol(cuda_bc2s_offsets_s1_0, o.offsets_s1_, sizeof(o.offsets_s1_));
    //   cudaMemcpyToSymbol(cuda_bc2s_offsets_s2_0, o.offsets_s2_, sizeof(o.offsets_s2_));

    // }
  }

  __device__ int
  cuda_bc2s_feature::distance(const bc2s& a, const i_short2& n, unsigned scale)
  {
    return bc2s_internals::distance_tex(a, n, scale);
    //return bc2s_internals::distance(*this, a, n, scale);
  }

  // void
  // bc2s_feature::bind()
  // {
  //   return bc2s_internals::distance_tex(a, n, scale);
  //   //return bc2s_internals::distance(*this, a, n, scale);
  // }

  __device__ bc2s
  cuda_bc2s_feature::operator()(const i_short2& p) const
  {
    //return bc2s_internals::compute_feature(*this, p);
    return bc2s_internals::compute_feature_tex(p);
  }

  // __device__ int
  // cuda_bc2s_feature::offsets_s1(int o) const
  // {
  //   assert(o < 8);
  //   i_int2 p(10,10);
  //   return &s1_(p + i_int2(circle_r3[o])) - &s1_(p);
  //   // return cuda_bc2s_offsets_s1_0[o];
  //   // switch (scaleid_)
  //   // {
  //   //   case 0: return cuda_bc2s_offsets_s1_0[o];
  //   //   case 1: return cuda_bc2s_offsets_s1_1[o];
  //   //   case 2: return cuda_bc2s_offsets_s1_2[o];
  //   // }
  //   //return cuda_bc2s_offsets_s1[o];
  // }

  // __device__ int
  // cuda_bc2s_feature::offsets_s2(int o) const
  // {
  //   assert(o < 8);
  //   i_int2 p(10,10);
  //   return &s2_(p + 2* i_int2(circle_r3[o])) - &s2_(p);
  //   // return cuda_bc2s_offsets_s2_0[o];
  //   // switch (scaleid_)
  //   // {
  //   //   case 0: return cuda_bc2s_offsets_s2_0[o];
  //   //   case 1: return cuda_bc2s_offsets_s2_1[o];
  //   //   case 2: return cuda_bc2s_offsets_s2_2[o];
  //   // }
  // }

#endif

  bc2s::bc2s()
  {
    ::memset(weights, 255, sizeof(weights));
  }

  bc2s::bc2s(const float4& o)
  {
    tex_float = o;
    ::memset(weights, 255, sizeof(weights));
  }

  bc2s::bc2s(const bc2s& o)
  {
    tex_float = o.tex_float;
    packed_weights = o.packed_weights;
  }

  bc2s&
  bc2s::operator=(const bc2s& o)
  {
    tex_float = o.tex_float;
    packed_weights = o.packed_weights;
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


  void
  bc2s::update_weights(const bc2s& f)
  {
    int best = 999;
    for (unsigned i = 0; i < 16; i++)
    {
      int d = ::abs(f[i] - distances[i]);
      if (d < best) best = d;
    }

    for (unsigned i = 0; i < 16; i++)
    {
      int d = ::abs(f[i] - distances[i]);
      //weights[i] = (weights[i] + 255.f / ((d - best)/1.f + 1)) / 2;
      weights[i] = 255.f / ((d - best)/5.f + 1);
      //weights[i] = 255.f / (d/5.f + 1);
      //weights[i] = 255;
    }
}

#endif
