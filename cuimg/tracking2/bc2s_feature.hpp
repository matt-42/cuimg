#ifndef CUIMG_BC2S_FEATURE_HPP_
# define CUIMG_BC2S_FEATURE_HPP_

# include <cuimg/architectures.h>
# include <cuimg/neighb2d_data.h>

namespace cuimg
{

  namespace bc2s_internals
  {
    template <typename O>
    __host__ __device__ inline int
    distance(const O& o, const bc2s& a, const i_short2& n)
    {
      int d = 0;

      const auto* data = &o.s1_(n);
      for(int i = 0; i < 8; i ++)
      {
	int v = data[o.offsets_s1_[i]].x;
	d += ::abs(v - a[i]);
      }

      data = &o.s2_(n);
      for(int i = 0; i < 8; i ++)
      {
      	int v = data[o.offsets_s2_[i]].x;
      	d += ::abs(v - a[8+i]);
      }

      // return d / (255.f * 16.f);
      return d;
    }

    template <typename O>
    __host__ __device__ inline bc2s
    compute_feature(const O& o, const i_int2& n)
    {
      bc2s b;

      const auto* data = &o.s1_(n);
      for(int i = 0; i < 8; i ++)
        b[i] = data[o.offsets_s1_[i]].x;

      data = &o.s2_(n);
      for(int i = 0; i < 8; i ++)
        b[i+8] = data[o.offsets_s2_[i]].x;

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

  }

  template <template <class> class I>
  bc2s_feature<I>::bc2s_feature(const obox2d& d)
    : s1_(d),
      s2_(d),
      tmp_(d)
  {
    for (unsigned i = 0; i < 16; i += 2)
    {
      point2d<int> p(10,10);
      offsets_s1_[i/2] = (long(&s1_(p + i_int2(circle_r3[i]))) - long(&s1_(p))) / sizeof(V);
      offsets_s2_[i/2] = (long(&s2_(p + i_int2(circle_r3[i])*2)) - long(&s2_(p))) / sizeof(V);
    }
  }


  template <template <class> class I>
  bc2s64_feature<I>::bc2s64_feature(const obox2d& d)
    : bc2s_feature<I>(d)
  {
    for (unsigned i = 0; i < 16; i += 4)
    {
      point2d<int> p(10,10);
      this->offsets_s1_[i/4] = (long(&this->s1_(p + i_int2(circle_r3[i]))) - long(&this->s1_(p))) / sizeof(V);
      this->offsets_s2_[i/4] = (long(&this->s2_(p + i_int2(circle_r3[i])*2)) - long(&this->s2_(p))) / sizeof(V);
    }
  }

  template <template <class> class I>
  void
  bc2s_feature<I>::update(const I<gl8u>& in)
  {
    dim3 dimblock = ::cuimg::dimblock(arch::cpu(), sizeof(V) + sizeof(i_uchar1), in.domain());

    local_jet_static_<0, 0, 2, 2>::run(in, s1_, tmp_, 0, dimblock);
    // local_jet_static_<0, 0, 3, 3>::run(in, s2_, tmp_, 0, dimblock);
    local_jet_static_<0, 0, 1, 1>::run(s1_, s2_, tmp_, 0, dimblock);
  }

  template <template <class> class I>
  int
  bc2s_feature<I>::distance(const bc2s& a, const i_short2& n)
  {
    return bc2s_internals::distance(*this, a, n);
  }

  template <template <class> class I>
  int
  bc2s64_feature<I>::distance(const bc2s64& a, const i_short2& n)
  {
    return bc2s_internals::distance64(*this, a, n);
  }

  template <template <class> class I>
  bc2s
  bc2s_feature<I>::operator()(const i_short2& p) const
  {
    return bc2s_internals::compute_feature(*this, p);
  }

  template <template <class> class I>
  bc2s64
  bc2s64_feature<I>::operator()(const i_short2& p) const
  {
    return bc2s_internals::compute_feature64(*this, p);
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
