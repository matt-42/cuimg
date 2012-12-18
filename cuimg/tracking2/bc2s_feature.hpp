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

  // template <template <class> class I>
  // bc2s_feature<I>::bc2s_feature(const bc2s_feature& f)
  // {
  //   *this = f;
  // }

  // template <template <class> class I>
  // bc2s_feature<I>&
  // bc2s_feature<I>::operator=(const bc2s_feature& f)
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
  bc2s
  bc2s_feature<I>::operator()(const i_short2& p) const
  {
    return bc2s_internals::compute_feature(*this, p);
  }


  // ########## bc2s feature vector methods. ###########
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


}

#endif
