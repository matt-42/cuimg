#ifndef TRAJECTORY_TRACER_H_
# define TRAJECTORY_TRACER_H_

# include <string>

# ifndef NO_CUDA
#  include <curand.h>
# endif

# include <cuimg/gpu/device_image2d.h>
# include <cuimg/image2d_target.h>

using namespace cuimg;


template <unsigned TG>
class trajectory_tracer
{
 public:
  enum { target = TG };

  typedef image2d_target(target, i_short2) image2d_s2;
  typedef image2d_target(target, i_float1) image2d_f1;
  typedef image2d_target(target, i_float4) image2d_f4;
  typedef image2d_target(target, char) image2d_c;

  typedef obox2d domain_t;

  trajectory_tracer(const domain_t& d);

  template <typename P>
  void update(const image2d_s2& matches,
              const device_image2d<P>& particles);

  void display(const std::string& w, const image2d_f1& colors);

  struct trace
  {
    __host__ __device__
    trace& operator=(const trace& tr)
    {
      color = tr.color;
      age = tr.age;
      for (int i = 0; i < 10; i++)
        history[i] = tr.history[i];
      return *this;
    }

    __host__ __device__ void reset()
    {
      color = i_float4(0.f, 0.f, 0.f, 1.f);
      age = 0;
      for (int i = 0; i < 10; i++)
        history[i] = i_short2(-1, -1);
    }

    i_float4 color;
    i_short2 history[10];
    int age;
    __host__ __device__
    bool is_valid() { return color != i_float4(0.f, 0.f, 0.f, 1.f); }

    __host__ __device__ void set_current_pos(i_short2 p) { history[age%10] = p; }

    __host__ __device__ i_short2 get_past_pos(int i)
    {
      if (age > i)
        return history[(age + 10 - i)%10];
      else
        return history[1];
    }
  };

  image2d_f4& straj() { return straj_; }

 private:

#ifndef NO_CUDA
  curandGenerator_t gen;
#endif

  image2d_f4 rand_colors_;
  image2d_f4 display_image_;
  image2d_f4 traj_;
  image2d_f4 straj_;
  image2d_f4 traj_heads_;
  device_image2d<trace> traces1_;
  device_image2d<trace> traces2_;
  device_image2d<trace>* traces_;
  device_image2d<trace>* new_traces_;
  device_image2d<short> age_;
};

# include <cuimg/tracking/trajectory_tracer.hpp>

#endif
