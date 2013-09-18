#ifndef CUIMG_BC2S_FEATURE_H_
# define CUIMG_BC2S_FEATURE_H_

# include <map>
# include <cuimg/architectures.h>
# include <cuimg/kernel_type.h>
# include <cuimg/gpu/cuda.h>
# include <cuimg/improved_builtin.h>
# include <cuimg/gl.h>
# include <cuimg/gpu/kernel_image2d.h>
# include <cuimg/gpu/gaussian_blur.h>

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

    inline __host__ __device__
    void update_weights(const bc2s& f);

    union
    {
      unsigned char weights[16];
      float4 packed_weights;
    };
  };

  struct bc2s_256
  {
    inline __host__ __device__ bc2s_256();
    inline __host__ __device__ bc2s_256(const bc2s_256& o);

    inline __host__ __device__
    bc2s_256& operator=(const bc2s_256& o);

    inline __host__ __device__
    const unsigned char& operator[](const unsigned i) const;

    inline __host__ __device__
    unsigned char& operator[](const unsigned i);

    union
    {
      unsigned char distances[32];
      float4 tex_float[2];
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

#ifndef NO_CUDA
  class cuda_bc2s_feature;
#endif

  template <typename A>
  class bc2s_feature
  {
  public:
    typedef bc2s value_type;
    typedef A architecture;

    typedef gl8u V;
    typedef typename A::template image2d<V>::ret I;
    bc2s_feature() {};
    bc2s_feature(const obox2d& o);
    bc2s_feature(const bc2s_feature<A>& f);
    bc2s_feature& operator=(const bc2s_feature<A>& f);

    inline const obox2d& domain() const { return s1_.domain(); }

#ifndef NO_CUDA
    void update(const I& in, const cuda_gpu&);
    void bind(const cuda_gpu&);
#endif

    void bind();
    void bind(const cpu&);
    //void unbind();
    void update(const I& in, const cpu&);
    int distance(const bc2s& a, const i_short2& n, unsigned scale = 1);
    bc2s operator()(const i_short2& p) const;

    const I& s1() const { return s1_; }
    const I& s2() const { return s2_; }

    inline int offsets_s1(int o) const;
    inline int offsets_s2(int o) const;

    inline int border_needed() const { return 6; }
    void swap(bc2s_feature<A>& o);
  public:
    I s1_;
    I s2_;
    I tmp_;
    int offsets_s1_[8];
    int offsets_s2_[8];
    gaussian_kernel<A> kernel_1_;
    gaussian_kernel<A> kernel_2_;
  };

  template <> struct kernel_type<bc2s_feature<cpu> > { typedef bc2s_feature<cpu>& ret; };

#ifndef NO_CUDA
  template <> struct kernel_type<bc2s_feature<cuda_gpu> > { typedef cuda_bc2s_feature ret; };

  // extern __constant__ int cuda_bc2s_offsets_s1_0[8];
  // extern __constant__ int cuda_bc2s_offsets_s2_0[8];

  // extern __constant__ int cuda_bc2s_offsets_s1_2[8];
  // extern __constant__ int cuda_bc2s_offsets_s2_2[8];

  // extern __constant__ int cuda_bc2s_offsets_s1_1[8];
  // extern __constant__ int cuda_bc2s_offsets_s2_1[8];

  class cuda_bc2s_feature
  {
  public:
    typedef cuda_gpu architecture;
    typedef bc2s value_type;
    typedef gl8u V;
    inline cuda_bc2s_feature(bc2s_feature<cuda_gpu>&);

    inline __device__ int distance(const bc2s& a, const i_short2& n, unsigned scale = 1);
    inline __device__ bc2s operator()(const i_short2& p) const;
    // inline __host__ __device__ int offsets_s1(int o) const;
    // inline __host__ __device__ int offsets_s2(int o) const;
    inline __host__ __device__ const kernel_image2d<V>& s1() const { return s1_; }
    inline __host__ __device__ const kernel_image2d<V>& s2() const { return s2_; }
    inline __host__ __device__ const obox2d& domain() const { return s1_.domain(); }

  public:
    kernel_image2d<V> s1_;
    kernel_image2d<V> s2_;
    // unsigned scaleid_;
    // static std::map<int, int> scales;
  };
#endif

  template <typename A>
  class bc2s_256_feature
  {
  public:
    typedef A architecture;
    typedef gl8u V;
    typedef typename A::template image2d<V>::ret I;
    typedef bc2s_256 value_type;
    bc2s_256_feature(const obox2d& o);
    bc2s_256_feature(const bc2s_256_feature<A>& f);
    bc2s_256_feature& operator=(const bc2s_256_feature<A>& f);

    inline const obox2d& domain() const { return s1_.domain(); }

    void update(const I& in, const cpu&);
    int distance(const bc2s_256& a, const i_short2& n, unsigned scale = 1);
    bc2s_256 operator()(const i_short2& p) const;

    const I& s1() const { return s1_; }
    const I& s2() const { return s2_; }

    inline int border_needed() const { return 6; }

    inline int offsets_s1(int o) const { return offsets_s1_[o]; }
    inline int offsets_s2(int o) const { return offsets_s2_[o]; }

    void swap(bc2s_256_feature<A>& o);
    void bind() {}

  public:
    I s1_;
    I s2_;
    I tmp_;
    int offsets_s1_[16];
    int offsets_s2_[16];
  };


  template <typename A>
  class bc2s64_feature : public bc2s_feature<A>
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

