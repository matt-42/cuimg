#ifndef CUIMG_MULTI_SCALE_MATCHER_HPP_
# define  CUIMG_MULTI_SCALE_MATCHER_HPP_

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
# include <cuimg/gpu/tracking/matcher_tools.h>


namespace cuimg
{


  __constant__ const int c49_s1[49][2] =
  {
    {-3, 3},  {-2, 3},  {-1,  3}, {0,  3}, {1,  3}, {2,  3}, {3,  3},
    {-3, 2},  {-2, 2},  {-1,  2}, {0,  2}, {1,  2}, {2,  2}, {3,  2},
    {-3, 1},  {-2, 1},  {-1,  1}, {0,  1}, {1,  1}, {2,  1}, {3,  1},
    {-3, 0},  {-2, 0},  {-1,  0}, {0,  0}, {1,  0}, {2,  0}, {3,  0},
    {-3, -1}, {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {2, -1}, {3, -1},
    {-3, -2}, {-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2}, {3, -2},
    {-3, -3}, {-2, -3}, {-1, -3}, {0, -3}, {1, -3}, {2, -3}, {3, -3}
  };

  __constant__ const int c49_s2[49][2] =
  {
    {-6, 6},  {-4, 6},  {-2,  6}, {0,  6}, {2,  6}, {4,  6}, {6,  6},
    {-6, 4},  {-4, 4},  {-2,  4}, {0,  4}, {2,  4}, {4,  4}, {6,  4},
    {-6, 2},  {-4, 2},  {-2,  2}, {0,  2}, {2,  2}, {4,  2}, {6,  2},
    {-6, 0},  {-4, 0},  {-2,  0}, {0,  0}, {2,  0}, {4,  0}, {6,  0},
    {-6, -2}, {-4, -2}, {-2, -2}, {0, -2}, {2, -2}, {4, -2}, {6, -2},
    {-6, -4}, {-4, -4}, {-2, -4}, {0, -4}, {2, -4}, {4, -4}, {6, -4},
    {-6, -6}, {-4, -6}, {-2, -6}, {0, -6}, {2, -6}, {4, -6}, {6, -6}
  };

  __constant__ const int c49_s3[49][2] =
  {
    {-12, 12},  {-8, 12},  {-4,  12}, {0,  12}, {4,  12}, {8,  12}, {12,  12},
    {-12, 8},  {-8, 8},  {-4,  8}, {0,  8}, {4,  8}, {8,  8}, {12,  8},
    {-12, 4},  {-8, 4},  {-4,  4}, {0,  4}, {4,  4}, {8,  4}, {12,  4},
    {-12, 0},  {-8, 0},  {-4,  0}, {0,  0}, {4,  0}, {8,  0}, {12,  0},
    {-12, -4}, {-8, -4}, {-4, -4}, {0, -4}, {4, -4}, {8, -4}, {12, -4},
    {-12, -8}, {-8, -8}, {-4, -8}, {0, -8}, {4, -8}, {8, -8}, {12, -8},
    {-12, -12}, {-8, -12}, {-4, -12}, {0, -12}, {4, -12}, {8, -12}, {12, -12}
  };

  __constant__ const int c49_s4[49][2] =
  {
    {-24, 24},  {-16, 24},  {-8,  24}, {0,  24}, {8,  24}, {16,  24}, {24,  24},
    {-24, 16},  {-16, 16},  {-8,  16}, {0,  16}, {8,  16}, {16,  16}, {24,  16},
    {-24, 8},  {-16, 8},  {-8,  8}, {0,  8}, {8,  8}, {16,  8}, {24,  8},
    {-24, 0},  {-16, 0},  {-8,  0}, {0,  0}, {8,  0}, {16,  0}, {24,  0},
    {-24, -8}, {-16, -8}, {-8, -8}, {0, -8}, {8, -8}, {16, -8}, {24, -8},
    {-24, -16}, {-16, -16}, {-8, -16}, {0, -16}, {8, -16}, {16, -16}, {24, -16},
    {-24, -24}, {-16, -24}, {-8, -24}, {0, -24}, {8, -24}, {16, -24}, {24, -24}
  };



  template <typename F, unsigned S>
  __device__ point2d<int> matching_search(F& f, const typename F::feature_t& p_state, point2d<int> prediction,
                                          float& match_distance)
  {
    point2d<int> match;
    for_all_in_static_neighb2d(prediction, n, c81) if (f.current_frame().has(n))
    {
      float d = f.distance(p_state, f.current_frame()(n));
      if (d < match_distance)
      {
        match = n;
        match_distance = d;
      }
    }
    return match;
  }


    template <typename F, unsigned S>
  __device__ point2d<int> matching_search_s2(F& f, const typename F::feature_t& p_state, point2d<int> prediction,
                                            float& match_distance)
  {
    point2d<int> match = prediction;
    for_all_in_static_neighb2d(prediction, n, c49_s3) if (f.current_frame().has(n))
    {
      float d = f.distance_s2(p_state, f.current_frame()(n));
      if (d < match_distance)
      {
        match = n;
        match_distance = d;
      }
    }
    return match;
  }

  template <typename F, typename T>
  __global__ void multi_scale_matching_kernel(F f,
                                        kernel_image2d<T> particles,
                                        kernel_image2d<T> new_particles,
                                        kernel_image2d<i_short2> matches,
                                        kernel_image2d<i_float4> dist,
                                        kernel_image2d<i_float4> test)
  {
    const float matching_threshold = 0.3f;
    point2d<int> p = thread_pos2d();
    if (!particles.has(p))
      return;

    test(p) = i_float4(1.f, 1.f, 0.f, 1.f);
    if (particles(p).age == 0)
    {
      dist(p) = i_float4(1.f, 0.f, 0.f, 1.f);
      return;
    }

    // Search for the best match in the neighborhood.
    int p_age = particles(p).age;
    typename F::feature_t p_state = particles(p).state;
    point2d<int> match = p;

    float match_distance = 999999.f;
    if (p_age == 1)
    //if (true)
    {
      match = matching_search<F, 81>(f, p_state, p, match_distance);
      if (match_distance > matching_threshold)
      {
        float m_s2 = 9999999.f;
        match = matching_search_s2<F, 49>(f, p_state, p, m_s2);
        match = matching_search<F, 81>(f, p_state, match, match_distance);
      }
    }
    else
    {
      point2d<int> prediction = i_int2(i_int2(p) + particles(p).speed + particles(p).acceleration);
      if (!f.current_frame().has(prediction))
      {
        dist(p) = i_float4(0.f, 1.f, 1.f, 1.f);
        return;
      }
      match = prediction;
      match_distance = f.distance(p_state, f.current_frame()(prediction));
      if (match_distance > 0.1f)
      {
        for_all_in_static_neighb2d(prediction, n, c49) if (particles.has(n))
        {
          float d = f.distance(p_state, f.current_frame()(n));// + norml2(i_int2(*n) - i_int2(prediction))/10.f;
          if (d < match_distance)
          {
            match = n;
            match_distance = d;
            if (match_distance < 0.1f) break;
          }
        }
      }
      if (match_distance > matching_threshold)
      {
        float m_s2 = 9999999.f;
        match = matching_search_s2<F, 49>(f, p_state, p, m_s2);
        match = matching_search<F, 81>(f, p_state, match, match_distance);
      }

    }


    if (match_distance <= matching_threshold// &&
        //new_particles(match).age <= (p_age + 1)
        )
    {

      i_float2 new_speed = i_int2(match) - i_int2(p);
      if (p_age == 1)
      {
        new_particles(match).speed = new_speed;
        new_particles(match).acceleration = i_float2(0.f, 0.f);
      }
      else
      {
        i_float2 s = (new_speed * 1.f + particles(p).speed) / 2.f;
        new_particles(match).acceleration = s - particles(p).speed;
        new_particles(match).speed = s;
      }
      //new_particles(match).state = f.current_frame()(match); //p_state;//

      //new_particles(match).state = (p_state*1.f + f.current_frame()(match)*3.f) / 4.f;
      new_particles(match).state = weighted_mean(p_state, 1.f, f.current_frame()(match), 1.f);

      new_particles(match).age = p_age + 1;
      particles(p) = new_particles(match);
      matches(p) = i_short2(match.row(), match.col());
      dist(p) = i_float4(match_distance, match_distance, match_distance, 1.f);
      //dist(p) = i_float4(0.f, 1.f, 0.f, 1.f);
    }
    else
    {

        matches(p) = i_short2(-1, -1);
        if (p_age == 1)
          dist(p) = i_float4(0.f, 0.f, 1.f, 1.f);
        else
          dist(p) = i_float4(1.f, 1.f, 0.f, 1.f);
    }

  }

  template <typename F>
  multi_scale_matcher<F>::multi_scale_matcher(const domain_t& d)
    : t1_(d),
      t2_(d),
      distance_(d),
      test_(d),
      test2_(d),
      matches_(d),
      frame_cpt(0)
  {
    particles_ = &t1_;
    new_particles_ = &t2_;

    particle p; p.age = 0;
    fill(*particles_, p);
    fill(*new_particles_, p);
  }

   template <typename F>
  void
  multi_scale_matcher<F>::swap_buffers()
  {
    std::swap(particles_, new_particles_);

    dim3 dimblock(16, 16, 1);
    dim3 dimgrid = grid_dimension(new_particles_->domain(), dimblock);

  //  particle p; p.age = 0;
    init_particles_kernel<particle><<<dimgrid, dimblock>>>(*new_particles_);
//    fill(*new_particles_, p);
  }

  template <typename F>
  void
  multi_scale_matcher<F>::update(F& f)
  {
    frame_cpt++;
    dim3 dimblock(16, 16, 1);
    dim3 dimgrid = grid_dimension(f.domain(), dimblock);




      //test_kernel<particle><<<dimgrid, dimblock>>>(*particles_, distance_);

      swap_buffers();
     multi_scale_matching_kernel<typename F::kernel_type, particle><<<dimgrid, dimblock>>>(f, *particles_, *new_particles_, matches_, distance_, test_);

      check_cuda_error();

    filter_robbers<particle><<<dimgrid, dimblock>>>(*particles_, *new_particles_, matches_);
    filter_robbers<particle><<<dimgrid, dimblock>>>(*particles_, *new_particles_, matches_);
    check_robbers<particle><<<dimgrid, dimblock>>>(*particles_, *new_particles_, matches_, test2_);

    if (!(frame_cpt % 2))
      create_particles_kernel<typename F::kernel_type, particle><<<dimgrid, dimblock>>>(f, *new_particles_, f.pertinence(), test2_);


#ifdef WITH_DISPLAY
      dg::widgets::ImageView("distances") <<= dg::dl() - distance_ - f.pertinence() - test_ - test2_;
#endif

  }

  template <typename F>
  const image2d<typename multi_scale_matcher<F>::particle>&
  multi_scale_matcher<F>::particles()
  {
    return *new_particles_;
  }

}

#endif // !  CUIMG_MULTI_SCALE_MATCHER_HPP_
