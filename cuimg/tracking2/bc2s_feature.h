#ifndef CUIMG_BC2S_FEATURE_H_
# define CUIMG_BC2S_FEATURE_H_

# include <cuimg/gpu/cuda.h>
# include <cuimg/improved_builtin.h>

namespace cuimg
{

  struct bc2s
  {
    inline __host__ __device__ bc2s();
    inline __host__ __device__ bc2s(const float4& o);
    inline __host__ __device__ bc2s(const bc2s& o);

    inline __host__ __device__
    bc2s& operator=(const bc2s& o);

    inline __host__ __device__
    const unsigned char& operator[](const unsigned i) const;

    inline __host__ __device__
    unsigned char& operator[](const unsigned i);

    union
    {
      unsigned char distances[16];
      float4 tex_float;
    };

  };

  template <template <class> class I>
  class bc2s_feature
  {
  public:
    typedef bc2s value_type;

    bc2s_feature(const obox2d& o);

    void update(const I<i_uchar1>& in);
    float distance(const bc2s& a, const i_short2& n);
    bc2s operator()(const i_short2& p) const;

  public:
    I<i_float1> s1_;
    I<i_float1> s2_;
    I<i_float1> tmp_;
    int offsets_s1_[8];
    int offsets_s2_[8];
  };

}

# include <cuimg/tracking2/bc2s_feature.hpp>

#endif

