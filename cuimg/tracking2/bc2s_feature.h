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


  struct bc2s64
  {
    inline __host__ __device__ bc2s64();
    //inline __host__ __device__ bc2s64(const float2& o);
    inline __host__ __device__ bc2s64(const bc2s64& o);

    inline __host__ __device__
    bc2s64& operator=(const bc2s64& o);

    inline __host__ __device__
    const unsigned char& operator[](const unsigned i) const;

    inline __host__ __device__
    unsigned char& operator[](const unsigned i);

    union
    {
      unsigned char distances[8];
      double tex_float;
      //float2 tex_float;
    };

  };

  template <template <class> class I>
  class bc2s_feature
  {
  public:
    typedef bc2s value_type;

    typedef gl8u V;
    bc2s_feature(const obox2d& o);
    bc2s_feature(const bc2s_feature& f);
    bc2s_feature& operator=(const bc2s_feature& f);

    inline const obox2d& domain() const { return s1_.domain(); }

    void update(const I<gl8u>& in);
    int distance(const bc2s& a, const i_short2& n, unsigned scale = 1);
    bc2s operator()(const i_short2& p) const;

    const I<V>& s1() const { return s1_; }
    const I<V>& s2() const { return s2_; }

  public:
    I<V> s1_;
    I<V> s2_;
    I<V> tmp_;
    int offsets_s1_[8];
    int offsets_s2_[8];
  };


  template <template <class> class I>
  class bc2s64_feature : public bc2s_feature<I>
  {
  public:
    typedef bc2s64 value_type;

    typedef gl8u V;
    bc2s64_feature(const obox2d& o);

    int distance(const bc2s64& a, const i_short2& n);
    bc2s64 operator()(const i_short2& p) const;
  };

}

# include <cuimg/tracking2/bc2s_feature.hpp>

#endif

