#ifndef CUIMG_MATCHER_TOOLS_H_
# define CUIMG_MATCHER_TOOLS_H_

# include <cuimg/gpu/kernel_image2d.h>

namespace cuimg
{

  template <typename T>
  __global__ void check_robbers( kernel_image2d<T> particles,
                                         kernel_image2d<T> new_particles,
                                         kernel_image2d<i_short2> matches,
                                         kernel_image2d<i_float4> test_)
  {
    point2d<int> p = thread_pos2d();
    if (!particles.has(p))
      return;


    if (particles(p).age == 0) test_(p) = i_float4(0.f, 0.f, 0.f, 1.f); // No particle here
    else if (!new_particles.has(matches(p))) test_(p) = i_float4(0.f, 0.f, 1.f, 1.f); // No match found
    else if (new_particles(matches(p)).age < particles(p).age) test_(p) = i_float4(1.f, 0.f, 0.f, 1.f);
    else test_(p) = i_float4(0.f, 1.f, 0.f, 1.f);
  }

  template <typename T>
  __global__ void filter_robbers(kernel_image2d<T> particles,
                                 kernel_image2d<T> new_particles,
                                 kernel_image2d<i_short2> matches)
  {
    point2d<int> p = thread_pos2d();
    if (!particles.has(p))
      return;

    if (particles(p).age == 0) return;

    if (new_particles.has(matches(p)) &&
        new_particles(matches(p)).age < particles(p).age)
    {
      point2d<int> match = i_int2(matches(p));
      new_particles(match) = particles(p);
    }

  }

  template <typename T>
  __global__ void init_particles_kernel(kernel_image2d<T> particles)
  {
    point2d<int> p = thread_pos2d();
    if (!particles.has(p) || particles(p).age == 0)
      return;
    particles(p).age = 0;
  }


  template <typename F, typename T>
  __global__ void create_particles_kernel(F f,
                                         kernel_image2d<T> particles,
                                         kernel_image2d<i_float1> pertinence,
                                         kernel_image2d<i_float4> test_)
  {
    point2d<int> p = thread_pos2d();
    if (!particles.has(p) || particles(p).age != 0)
      return;

    if (pertinence(p).x > 0.20f)
    {
      particles(p).age = 1;
      particles(p).state = f.current_frame()(p);
      particles(p).speed = i_float2(0.f, 0.f);
      particles(p).acceleration = i_float2(0.f, 0.f);
   //   test_(p) = i_float4(0.f, 1.f, 0.f, 1.f);
    }
   // else
  //    test_(p) = i_float4(1.f, 0.f, 0.f, 1.f);

  }

}

#endif // ! CUIMG_MATCHER_TOOLS_H_
