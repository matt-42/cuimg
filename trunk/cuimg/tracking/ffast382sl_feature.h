#ifndef CUIMG_FFAST382SL_FEATURE_H_
# define CUIMG_FFAST382SL_FEATURE_H_

# include <cuimg/gpu/device_image2d.h>
# include <cuimg/gpu/kernel_image2d.h>
# include <cuimg/point2d.h>
# include <cuimg/gl.h>
# include <cuimg/obox2d.h>
# include <cuimg/improved_builtin.h>
# include <cuimg/target.h>
# include <cuimg/image2d_target.h>

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
    const gl8u& operator[](const unsigned i) const
    {
      return *(const gl8u*)&(distances[i]);
    }

    __host__ __device__
    gl8u& operator[](const unsigned i)
    {
      return *(gl8u*)&(distances[i]);
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

  template <typename V>
  class kernel_ffast382sl_feature;

  template <typename V, unsigned T>
  class ffast382sl_feature
  {
  public:
    enum { target = T };

    typedef V vtype;
    /* typedef i_float1 V; */
    typedef image2d_target(target, i_uchar3) image2d_uc3;
    typedef image2d_target(target, i_float1) image2d_f1;
    typedef image2d_target(target, V) image2d_V;
    typedef image2d_target(target, i_float4) image2d_f4;
    typedef image2d_target(target, char) image2d_c;


    typedef dffast382sl feature_t;
    typedef image2d_target(target, feature_t) image2d_D;

    typedef obox2d domain_t;

    typedef kernel_ffast382sl_feature<V> kernel_type; // client side GPU type.

    inline ffast382sl_feature(const domain_t& d);

    inline void update(const image2d_V& in, const image2d_V& in_s2);

    inline const domain_t& domain() const;

    inline image2d_D& previous_frame();
    inline image2d_D& current_frame();
    inline image2d_f1& pertinence();
    inline image2d_V& s1();
    inline image2d_V& s2();

    const image2d_f4& feature_color() const;

    void display() const;

    void set_detector_thresh(float thr);

  private:
    inline void swap_buffers();

    image2d_V gl_frame_;
    image2d_V blurred_s1_;
    image2d_V blurred_s2_;
    image2d_V tmp_;

    image2d_f1 pertinence_;
    image2d_f1 pertinence2_;

    image2d_D f1_;
    image2d_D f2_;

    image2d_D* f_prev_;
    image2d_D* f_;

    image2d_f4 ffast382sl_color_;

    image2d_f4 color_blurred_;
    image2d_f4 color_tmp_;

    float grad_thresh_;

    cudaStream_t cuda_stream_;
    int frame_cpt_;
  };

  template <typename V>
  class kernel_ffast382sl_feature
  {
  public:
    typedef dffast382sl feature_t;

    template <unsigned target>
    __host__ __device__
    inline kernel_ffast382sl_feature(ffast382sl_feature<V, target>& f);

    inline
    __host__ __device__ float distance(const point2d<int>& p_prev,
                              const point2d<int>& p_cur);

    inline
    __host__ __device__ float distance_linear(const dffast382sl& a, const dffast382sl& b);

    inline
    __host__ __device__ float distance_linear(const dffast382sl& a,
					      const point2d<int>& n);

    /* inline */
    /* __host__ __device__ float distance_linear(const point2d<int>& a, */
    /*                                  const point2d<int>& b); */

    inline
    __host__ __device__ float distance_linear_s2(const dffast382sl& a, const dffast382sl& b);
    inline
    __host__ __device__ float distance(const dffast382sl& a, const dffast382sl& b);
    inline
    __host__ __device__ float distance_s2(const dffast382sl& a, const dffast382sl& b);

    __host__ __device__ inline
    dffast382sl weighted_mean(const dffast382sl& a, float aw, const point2d<int>& b, float bw);

    inline  __host__ __device__ dffast382sl
    new_state(const point2d<int>& n);

    inline __host__ __device__
    kernel_image2d<V>& s1();
    inline __host__ __device__
    kernel_image2d<V>& s2();

    __host__ __device__ inline
    kernel_image2d<i_float1>& pertinence();

  private:
    kernel_image2d<i_float1> pertinence_;
    kernel_image2d<V> s1_;
    kernel_image2d<V> s2_;
  };

}

# include <cuimg/tracking/ffast382sl_feature.hpp>

#endif // ! CUIMG_FFAST382SL_FEATURE_H_
