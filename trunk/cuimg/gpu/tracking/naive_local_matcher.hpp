#ifndef CUIMG_NAIVE_LOCAL_MATCHER_HPP_
# define  CUIMG_NAIVE_LOCAL_MATCHER_HPP_

#include <GL/glew.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <host_defines.h>
#include <cudaGL.h>
#include <cuda_gl_interop.h>



#include <dige/window.h>
#include <dige/widgets/image_view.h>
#include <dige/image.h>
#include <dige/process_events.h>
#include <dige/pick_coords.h>
#include <dige/widgets/slider.h>
#include <dige/widgets/push_button.h>
#include <dige/event/wait_event.h>
#include <dige/event/event_queue.h>
#include <dige/event/click.h>
#include <dige/event/key_release.h>
#include <dige/event/keycode.h>
#include <dige/recorder.h>
#include <cuimg/dige.h>


# include <cuimg/gpu/image2d.h>
# include <cuimg/gpu/fill.h>
# include <cuimg/neighb2d_data.h>
# include <cuimg/neighb_iterator2d.h>
# include <cuimg/static_neighb2d.h>
# include <cuimg/point2d.h>


namespace cuimg
{

  template <typename F, typename T>
  __global__ void naive_matching_kernel(F f,
                                        kernel_image2d<T> particles,
                                        kernel_image2d<T> new_particles,
                                        kernel_image2d<i_float4> dist)
  {
    point2d<int> p = thread_pos2d();
    if (!particles.has(p))
      return;
    if (particles(p).age == 0)
    {
      dist(p) = i_float4(1.f, 0.f, 0.f, 1.f);
      return;
    }


    // Search for the best match in the neighborhood.
    int p_age = particles(p).age;
    typename F::feature_t p_state = particles(p).state;
    point2d<int> match = p;
    
    float match_distance;
    if (p_age == 1)
    {
     match_distance = f.distance(p_state, f.current_frame()(p));
      for_all_in_static_neighb2d(p, n, c80) if (particles.has(n))
      {
        float d = f.distance(p_state, f.current_frame()(n));
        if (d < match_distance)
        {
          match = n;
          match_distance = d;
        }
      }
    }
    else
    {
      point2d<int> prediction = i_int2(i_int2(p) + particles(p).speed);
      if (!f.current_frame().has(prediction)) return;

       match_distance = f.distance(p_state, f.current_frame()(prediction));
      for_all_in_static_neighb2d(prediction, n, c49) if (particles.has(n))
      {
        float d = f.distance(p_state, f.current_frame()(n));// + norml2(i_int2(*n) - i_int2(prediction))/10.f;
        if (d < match_distance)
        {
          match = n;
          match_distance = d;
        }
      }
    }

    dist(p) = i_float4(match_distance, match_distance, match_distance, 1.f);

    if (match_distance < 0.5f && new_particles(match).age <= (p_age + 1))
    {

      i_float2 new_speed = i_int2(match) - i_int2(p); 
      if (p_age == 1)
        new_particles(match).speed = new_speed;
      else
        new_particles(match).speed = (new_speed * 1.f + particles(p).speed) / 2.f;
      new_particles(match).state = p_state;//f.current_frame()(match); //
      new_particles(match).age = p_age + 1;
    }
    else
    {
      dist(p) = i_float4(1.f, 1.f, 0.f, 1.f);
    }

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

    if (pertinence(p).x >= 0.1f)
    {
      particles(p).age = 1;
      particles(p).state = f.current_frame()(p);
      test_(p) = i_float4(0.f, 1.f, 0.f, 1.f);
    }
    else
      test_(p) = i_float4(1.f, 0.f, 0.f, 1.f);

  }

  template <typename F>
  naive_local_matcher<F>::naive_local_matcher(const domain_t& d)
    : t1_(d),
      t2_(d),
      distance_(d),
      test_(d),
      frame_cpt(0)
  {
    particles_ = &t1_;
    new_particles_ = &t2_;
  }

   template <typename F>
  void
  naive_local_matcher<F>::swap_buffers()
  {
    std::swap(particles_, new_particles_);
    particle p;
    fill(*new_particles_, p);
  }

  template <typename F>
  void
  naive_local_matcher<F>::update(F& f)
  {
    frame_cpt++;
    dim3 dimblock(16, 16, 1);
    dim3 dimgrid = grid_dimension(f.domain(), dimblock);

    swap_buffers();
    naive_matching_kernel<typename F::kernel_type, particle><<<dimgrid, dimblock>>>(f, *particles_, *new_particles_, distance_);
    if (!(frame_cpt % 90))
      create_particles_kernel<typename F::kernel_type, particle><<<dimgrid, dimblock>>>(f, *new_particles_, f.pertinence(), test_);
    check_cuda_error();

    dg::widgets::ImageView("distances") <<= dg::dl() - distance_ - f.pertinence() - test_;

  }

  template <typename F>
  const image2d<typename naive_local_matcher<F>::particle>&
  naive_local_matcher<F>::particles()
  {
    return *particles_;
  }

}

#endif // !  CUIMG_NAIVE_LOCAL_MATCHER_HPP_
