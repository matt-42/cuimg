#ifndef TRAJECTORY_TRACER_H_
# define TRAJECTORY_TRACER_H_

# include <string>
# include <curand.h>
# include <cuimg/gpu/image2d.h>

using namespace cuimg;

class trajectory_tracer
{
 public:
  typedef obox2d<point2d<int> > domain_t;

  trajectory_tracer(const domain_t& d);

  template <typename P>
  void update(const image2d<i_short2>& matches,
              const image2d<P>& particles);

  void display(const std::string& w, const image2d<i_float1>& colors);

  struct trace
  {
    i_float4 color;
    __host__ __device__
    bool is_valid() { return color != i_float4(0.f, 0.f, 0.f, 1.f); }
  };

 private:

  curandGenerator_t gen;
  image2d<i_float4> rand_colors_;
  image2d<i_float4> display_image_;
  image2d<i_float4> traj_;
  image2d<i_float4> traj_heads_;
  image2d<trace> traces1_;
  image2d<trace> traces2_;
  image2d<trace>* traces_;
  image2d<trace>* new_traces_;
  image2d<short> age_;
};

# include <cuimg/gpu/tracking/trajectory_tracer.hpp>

#endif
