#ifndef CUIMG_FAST382SL_FEATURE_H_
# define CUIMG_FAST382SL_FEATURE_H_

# include <cuimg/gpu/device_image2d.h>
# include <cuimg/gpu/kernel_image2d.h>
# include <cuimg/point2d.h>
# include <cuimg/obox2d.h>
# include <cuimg/gl.h>
# include <cuimg/improved_builtin.h>
# include <cuimg/target.h>
# include <cuimg/image2d_target.h>

namespace cuimg
{

  struct dfast382sl
  {
    __host__ __device__ dfast382sl() {}

    __host__ __device__ dfast382sl(const float4& o)
    {
      tex_float = o;
    }

    __host__ __device__ dfast382sl(const dfast382sl& o)
    {
      //pertinence = o.pertinence;
      tex_float = o.tex_float;
      //for (unsigned i = 0; i < 16; i++) distances[i] = o.distances[i];
    }

    __host__ __device__
    dfast382sl& operator=(const dfast382sl& o)
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
  dfast382sl operator+(const dfast382sl& a, const dfast382sl& b);
  __host__ __device__ inline
  dfast382sl operator-(const dfast382sl& a, const dfast382sl& b);

  template <typename S>
  __host__ __device__ inline
  dfast382sl operator/(const dfast382sl& a, const S& s);

  template <typename S>
  __host__ __device__ inline
  dfast382sl operator*(const dfast382sl& a, const S& s);

  class kernel_fast382sl_feature;

  template <target T>
  class fast382sl_feature
  {
  public:
    static const cuimg::target target = T;

    typedef gl01f vtype;
    typedef image2d_target(target, i_uchar3) image2d_uc3;
    typedef image2d_target(target, gl01f) image2d_f1;
    typedef image2d_target(target, i_float4) image2d_f4;
    typedef image2d_target(target, char) image2d_c;


    typedef dfast382sl feature_t;
    typedef image2d_target(target, feature_t) image2d_D;

    typedef obox2d domain_t;

    typedef kernel_fast382sl_feature kernel_type; // client side GPU type.

    inline fast382sl_feature(const domain_t& d);

    inline void update(const image2d_f1& in, const image2d_f1& in_s2);

    inline const domain_t& domain() const;

    inline image2d_D& previous_frame();
    inline image2d_D& current_frame();
    inline image2d_f1& pertinence();
    inline const image2d_f1& pertinence() const;
    inline image2d_f1& s1();
    inline image2d_f1& s2();

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

    image2d_f4 fast382sl_color_;

    image2d_f4 color_blurred_;
    image2d_f4 color_tmp_;

    float grad_thresh;

    cudaStream_t cuda_stream_;
    int frame_cpt_;
  };

  class kernel_fast382sl_feature
  {
  public:
    typedef dfast382sl feature_t;

    template <target target>
    __host__ __device__
    inline kernel_fast382sl_feature(fast382sl_feature<target>& f);


    inline
    __host__ __device__ float distance(const point2d<int>& p_prev,
                              const point2d<int>& p_cur);

    inline
    __host__ __device__ float distance_linear(const dfast382sl& a, const dfast382sl& b);

    inline
    __host__ __device__ float distance_linear(const dfast382sl& a,
					      const point2d<int>& n);

      inline
    __host__ __device__ float distance_linear_fast(const dfast382sl& a,
                                                   const point2d<int>& n);

    /* inline */
    /* __host__ __device__ float distance_linear(const point2d<int>& a, */
    /*                                  const point2d<int>& b); */

    inline
    __host__ __device__ float distance_linear_s2(const dfast382sl& a, const dfast382sl& b);
    inline
    __host__ __device__ float distance(const dfast382sl& a, const dfast382sl& b);
    inline
    __host__ __device__ float distance_s2(const dfast382sl& a, const dfast382sl& b);

    __host__ __device__ inline
    dfast382sl weighted_mean(const dfast382sl& a, float aw, const point2d<int>& b, float bw);

    inline  __host__ __device__ dfast382sl
    new_state(const point2d<int>& n);

    inline __host__ __device__
    kernel_image2d<i_float1>& s1();
    inline __host__ __device__
    kernel_image2d<i_float1>& s2();

    /* __host__ __device__ inline */
    /* kernel_image2d<dfast382sl>& previous_frame(); */
    /* __host__ __device__ inline */
    /* kernel_image2d<dfast382sl>& current_frame(); */
    __host__ __device__ inline
    kernel_image2d<i_float1>& pertinence();

  private:
    kernel_image2d<i_float1> pertinence_;
    kernel_image2d<dfast382sl> f_prev_;
    /* kernel_image2d<dfast382sl> f_; */
    kernel_image2d<i_float1> s1_;
    kernel_image2d<i_float1> s2_;
    int offsets_s1[16];
    int offsets_s2[16];
  };

}

# include <cuimg/tracking/fast382sl_feature.hpp>

#endif // ! CUIMG_FAST382SL_FEATURE_H_
