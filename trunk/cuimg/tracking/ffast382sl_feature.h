#ifndef CUIMG_FFAST382SL_FEATURE_H_
# define CUIMG_FFAST382SL_FEATURE_H_

# include <cuimg/gpu/image2d.h>
# include <cuimg/gpu/kernel_image2d.h>
# include <cuimg/point2d.h>
# include <cuimg/obox2d.h>
# include <cuimg/improved_builtin.h>

namespace cuimg
{

  struct dffast382sl
  {
    __host__ __device__ dffast382sl() {}

    __host__ __device__ dffast382sl(const float4& o)
    {
      tex_float = o;
    }

    __host__ __device__ dffast382sl(const dffast382sl& o)
    {
      //pertinence = o.pertinence;
      tex_float = o.tex_float;
      //for (unsigned i = 0; i < 16; i++) distances[i] = o.distances[i];
    }

    __host__ __device__
    dffast382sl& operator=(const dffast382sl& o)
    {
      //pertinence = o.pertinence;
      tex_float = o.tex_float;
      //for (unsigned i = 0; i < 16; i++) distances[i] = o.distances[i];
      return *this;
    }

    __host__ __device__
    const unsigned char& operator[](const unsigned i) const
    {
      return distances[i];
    }

    __host__ __device__
    unsigned char& operator[](const unsigned i)
    {
      return distances[i];
    }

    union
    {
      unsigned char distances[16];
      float4 tex_float;
    };

  };

  __host__ __device__ inline
  dffast382sl operator+(const dffast382sl& a, const dffast382sl& b);
  __host__ __device__ inline
  dffast382sl operator-(const dffast382sl& a, const dffast382sl& b);

  template <typename S>
  __host__ __device__ inline
  dffast382sl operator/(const dffast382sl& a, const S& s);

  template <typename S>
  __host__ __device__ inline
  dffast382sl operator*(const dffast382sl& a, const S& s);

  class kernel_ffast382sl_feature;

  class ffast382sl_feature
  {
  public:
    typedef dffast382sl feature_t;
    typedef obox2d<point2d<int> > domain_t;

    typedef kernel_ffast382sl_feature kernel_type;

    inline ffast382sl_feature(const domain_t& d);

    inline void update(const image2d_f4& in);
    inline void update(const image2d_f1& in, const image2d_f1& in_s2);

    inline const domain_t& domain() const;

    inline image2d_D& previous_frame();
    inline image2d_D& current_frame();
    inline image2d_f1& pertinence();

    const image2d_f4& feature_color() const;

    void display() const;

  private:
    inline void swap_buffers();


    image2d_f1 gl_frame_;
    image2d_f1 blurred_s1_;
    image2d_f1 blurred_s2_;
    image2d_f1 tmp_;

    image2d_f1 pertinence_;
    image2d_f1 pertinence2_;

    image2d_D f1_;
    image2d_D f2_;

    image2d_D* f_prev_;
    image2d_D* f_;

    image2d_f4 ffast382sl_color_;

    image2d_f4 color_blurred_;
    image2d_f4 color_tmp_;

    float grad_thresh;

    cudaStream_t cuda_stream_;
  };

  class kernel_ffast382sl_feature
  {
  public:
    typedef dffast382sl feature_t;

    inline kernel_ffast382sl_feature(ffast382sl_feature& f);


    inline
    __device__ float distance(const point2d<int>& p_prev,
                              const point2d<int>& p_cur);

    inline
    __device__ float distance_linear(const dffast382sl& a, const dffast382sl& b);

    inline
    __device__ float distance_linear(const dffast382sl& a,
                                     const point2d<int>& n);

    inline
    __device__ float distance_linear(const point2d<int>& a,
                                     const point2d<int>& b);

    inline
    __device__ float distance_linear_s2(const dffast382sl& a, const dffast382sl& b);
    inline
    __device__ float distance(const dffast382sl& a, const dffast382sl& b);
    inline
    __device__ float distance_s2(const dffast382sl& a, const dffast382sl& b);

    __device__ inline
    dffast382sl weighted_mean(const dffast382sl& a, float aw, const point2d<int>& b, float bw);

    inline  __device__ dffast382sl
    new_state(const point2d<int>& n);

    /* __device__ inline */
    /* kernel_image2d<dffast382sl>& previous_frame(); */
    __device__ inline
    kernel_image2d<dffast382sl>& current_frame();
    __device__ inline
    kernel_image2d<i_float1>& pertinence();

  private:
    kernel_image2d<i_float1> pertinence_;
    kernel_image2d<dffast382sl> f_prev_;
    kernel_image2d<dffast382sl> f_;
  };

}

# include <cuimg/tracking/ffast382sl_feature.hpp>

#endif // ! CUIMG_FFAST382SL_FEATURE_H_
