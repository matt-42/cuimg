#ifndef CUIMG_NAIVE_LOCAL_MATCHER_THRUST_HPP_
# define  CUIMG_NAIVE_LOCAL_MATCHER_THRUST_HPP_

#include <GL/glew.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <host_defines.h>
#include <cudaGL.h>
#include <cuda_gl_interop.h>

# include <thrust/remove.h>



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


# include <cuimg/gpu/texture.h>
# include <cuimg/gpu/image2d.h>
# include <cuimg/gpu/fill.h>
# include <cuimg/neighb2d_data.h>
# include <cuimg/neighb_iterator2d.h>
# include <cuimg/static_neighb2d.h>
# include <cuimg/point2d.h>

# include <cuimg/tracking/matcher_tools.h>


namespace cuimg
{

  // #define feat_tex UNIT_STATIC(feat_tex_id)

  // texture<float4, 2, cudaReadModeElementType> feat_tex;

  template <typename T>
  __global__ void init_particles_vec(i_short2* particles_vec,
                                     kernel_image2d<T> particles)
  {
    point2d<int> p = thread_pos2d();

    if (!particles.has(p)) return;

    i_short2 res(-1, -1);
    if (particles(p).age != 0)
      res = i_short2(p.row(), p.col());
    particles_vec[p.row() * particles.ncols() + p.col()] = res;
  }

  template <typename F, typename T>
  __global__ void naive_matching_kernel2(i_short2* particles_vec,
                                         unsigned n_particles,
                                         F f,
                                         kernel_image2d<T> particles,
                                         kernel_image2d<T> new_particles,
                                         kernel_image2d<i_short2> matches,
                                         const kernel_image2d<i_short2> ls_matches,
                                         i_short2 mvt,
                                         T* compact_particles
#ifdef WITH_DISPLAY
                                         ,kernel_image2d<i_float4> dist,
                                         int age_filter
#endif
                                         )
  {
    unsigned threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid >= n_particles) return;
    point2d<int> p = particles_vec[threadid];

    //point2d<int> p = thread_pos2d();
    if (!particles.has(p))
      return;

    matches(p) = i_short2(-1, -1);

    if (p.row() < 6 || p.row() >= particles.domain().nrows() - 6 ||
        p.col() < 6 || p.col() >= particles.domain().ncols() - 6)
    {
      new_particles(p).age = 0;
      return;
    }

    if (particles(p).age == 0)
    {
#ifdef WITH_DISPLAY
      dist(p) = i_float4(0.2f, 0.f, 0.f, 1.f);
#endif
      return;
    }

    // Search for the best match in the neighborhood.
    short p_age = particles(p).age;
    typename F::feature_t p_state = particles(p).state;
    //float4 p_state = particles(p).state.tex_float;

    point2d<int> match = p;

    bool ls_p_found = false;

    float match_distance;
    point2d<int> prediction = i_int2(p);
    if (p_age == 1)
      //if (true)
    {
      point2d<int> lsp(p.row() / 2, p.col() / 2);
      point2d<int> ls_pred;
      {for_all_in_static_neighb2d(lsp, n, c25)
        {
          if (ls_matches(n).x != -1)
          {
            ls_pred = point2d<int>(ls_matches(n).x * 2,
                                   ls_matches(n).y * 2);
            ls_p_found = true;
          }
        }
      }

      match = p;

      //match_distance = f.distance_linear(p_state, tex2D(feat_tex, p));
      typename F::feature_t vnp = f.current_frame()(p);
      match_distance = f.distance_linear(p_state, vnp);

      //if (match_distance > 0.01f)
      if (true)
      {
        if (ls_p_found)
          prediction = ls_pred;
        else
          prediction = i_int2(p) + mvt;

        if (!f.current_frame().has(prediction)
            ||
            prediction.row() < 6 || prediction.row() >= particles.domain().nrows() - 6 ||
            prediction.col() < 6 || prediction.col() >= particles.domain().ncols() - 6
            ) return;
        // if (!particles.has(prediction)) return;
        /*
          for_all_in_static_neighb2d(prediction, n, c81) if (particles.has(n))
          {
          //float d = f.distance_linear(p_state, tex2D(feat_tex, n));
          float d = f.distance_linear(p_state, f.current_frame()(n));
          if (d < match_distance)
          {
          match = n;
          match_distance = d;
          //if (match_distance < 0.1f) break;
          }
          }*/
      }
    }
    else
    {
      //ls_p_found = false;
      // if (ls_p_found)
      //   prediction = ls_pred;
      // else
      // prediction = i_int2(i_int2(p) + particles(p).speed + particles(p).acceleration + mvt);

      prediction = i_int2(i_int2(p) + particles(p).speed + mvt);

      if (!f.current_frame().has(prediction)
          ||
          prediction.row() < 6 || prediction.row() >= particles.domain().nrows() - 6 ||
          prediction.col() < 6 || prediction.col() >= particles.domain().ncols() - 6
          ) return;

      match = prediction;

      //match_distance = f.distance_linear(p_state, tex2D(feat_tex, prediction));
      match_distance = f.distance_linear(p_state, f.current_frame()(prediction));
    }

    //    if (match_distance > 0.01f)
    {

      unsigned match_i = 8;
      for (int search = 0; search < 7; search++)
      {
        int i = c8_it[match_i][0];

        {
          point2d<int> n(prediction.row() + c8[i][0],
                         prediction.col() + c8[i][1]);
          {
            typename F::feature_t vn = f.current_frame()(n);
            float d = f.distance_linear(p_state, vn);
            if (d < match_distance)
            {
              match = n;
              match_i = i;
              match_distance = d;
            }
          }
          i = (i + 1) & 7;
        }

#pragma unroll 4
        for(; i != c8_it[match_i][1]; i = (i + 1) & 7)
        {
          point2d<int> n(prediction.row() + c8[i][0],
                         prediction.col() + c8[i][1]);
          {
            typename F::feature_t vn = f.current_frame()(n);
            float d = f.distance_linear(p_state, vn);
            if (d < match_distance)
            {
              match = n;
              match_i = i;
              match_distance = d;
            }
          }
        }


        if (i_int2(prediction) == i_int2(match) ||
            match.row() < 6 || match.row() >= (particles.domain().nrows() - 6) ||
            match.col() < 6 || match.col() >= (particles.domain().ncols() - 6)
            )
          break;
        else
          prediction = match;
      }

    }
    // for_all_in_static_neighb2d(prediction, n, c49) if (particles.has(n))
    // {
    //   //float d = f.distance_linear(p_state, tex2D(feat_tex, n));
    //   float d = f.distance_linear(p_state, f.current_frame()(n));
    //   if (d < match_distance)
    //   {
    //     match = n;
    //     match_distance = d;
    //     //if (match_distance < 0.1f) break;
    //   }
    // }
#ifdef WITH_DISPLAY
    dist(p) = i_float4(match_distance, match_distance, match_distance, 1.f);
#endif

    char fault = particles(p).fault;
    if (f.pertinence()(match) < 0.15f// ||
        //::abs(f.pertinence()(match) -
        //      particles(p).pertinence) > 0.2f
        )
      fault++;
    else
      fault = 0;

    if (f.pertinence()(match) < 0.001f) fault = 100;

    if (match_distance < (0.1f) && fault < 10)
    {

      i_float2 new_speed = i_int2(match) - i_int2(p);
      if (p_age == 1)
      {
        new_particles(match).speed = new_speed;
        //new_particles(match).acceleration = i_float2(0.f, 0.f);
        new_particles(match).brut_acceleration = i_float2(0.f, 0.f);
      }
      else
      {
        //i_float2 s =
        //i_float2 s = new_speed;
        //new_particles(match).acceleration = s - particles(p).speed;
        new_particles(match).brut_acceleration =  new_speed - particles(p).speed;

        new_particles(match).speed = (new_speed * 3.f + particles(p).speed) / 4.f;

      }
      //new_particles(match).state = f.current_frame()(match); //p_state;//


      if (!(p_age % 50))
        new_particles(match).ipos = i_int2(p);
      else
        new_particles(match).ipos = particles(p).ipos;

      new_particles(match).state =
        weighted_mean(p_state, 1.f,
                      f.current_frame()(match), 3.f);
      // new_particles(match).state = (p_state*1.f + f.current_frame()(match)*3.f) / 4.f;
      new_particles(match).age = p_age + 1;
      //new_particles(match).pertinence = f.pertinence()(match);
      new_particles(match).fault = fault;
      particles(p) = new_particles(match);
      compact_particles[threadid] = new_particles(match);

      matches(p) = i_short2(match.row(), match.col());

      // if (ls_p_found)
      //   dist(p) = i_float4(0.f, 0.7f, 0.f, 1.f);

    }
    else
    {
      matches(p) = i_short2(-1, -1);

#ifdef WITH_DISPLAY
      if (p_age <= 5)
        dist(p) = i_float4(1.f, 1.f, 1.f, 1.f);


      // if (fault >= 5)
      //   dist(p) = i_float4(0.f, 1.f, 0.f, 1.f);
      // else
      // if (match_distance > 1000.f)
      //   dist(p) = i_float4(1.f, 0.f, 0.f, 1.f);
      // else
      //   dist(p) = i_float4(1.f, 1.f, 0.f, 1.f);
#endif
    }

#ifdef WITH_DISPLAY
    if (particles(p).age > age_filter)
    {
      dist(p) = i_float4(0.f, 0.f, 0.f, 0.f);
    }
    //dist(p) = p_state.o1;
#endif

  }

  template <typename F>
  naive_local_matcher_thrust<F>::naive_local_matcher_thrust(const domain_t& d)
    : t1_(d),
      t2_(d),
      distance_(d),
      test_(d),
      test2_(d),
      matches_(d),
      errors_(d),
      fm_disp_(d),
      particles_vec1_(d.nrows() * d.ncols()),
      particles_vec2_(d.nrows() * d.ncols()),
      compact_particles_(d.nrows() * d.ncols()),
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
  naive_local_matcher_thrust<F>::swap_buffers()
  {
    std::swap(particles_, new_particles_);

    dim3 dimblock(128, 1, 1);
    dim3 dimgrid = grid_dimension(new_particles_->domain(), dimblock);

    cudaMemset2D(new_particles_->data(), new_particles_->pitch(), 0,
                 new_particles_->pitch(), new_particles_->nrows());
  //  particle p; p.age = 0;
    // init_particles_kernel<particle><<<dimgrid, dimblock>>>(*new_particles_);
//    fill(*new_particles_, p);
  }


  template<typename ForwardIterator,
           typename T>
  ForwardIterator remove(ForwardIterator first,
                            ForwardIterator last,
                           // thrust::detail::uninitialized_array<
                           //    typename thrust::iterator_traits<ForwardIterator>::value_type,
                           //    typename thrust::iterator_space<ForwardIterator>::type
                           //    >&
                         thrust::device_vector<i_short2>&
                         temp,
                            const T& value)
{
  typedef typename thrust::iterator_traits<ForwardIterator>::value_type InputType;
  typedef typename thrust::iterator_space<ForwardIterator>::type Space;

  thrust::detail::equal_to_value<T> pred(value);

  // remove into temp
  return thrust::detail::backend::generic::copy_if(temp.begin(), temp.end(), temp.begin(), first,
                                                          thrust::detail::not1(pred));
}

  template <typename F>
  unsigned
  naive_local_matcher_thrust<F>::n_particles() const
  {
    return n_particles_;
  }

  template <typename F>
  void
  naive_local_matcher_thrust<F>::update(F& f, global_mvt_thread<naive_local_matcher_thrust<F> >& t_mvt,
                                 const image2d_s2& ls_matches)
  {
    frame_cpt++;
    dim3 dimblock(128, 1, 1);
    dim3 dimgrid = grid_dimension(f.domain(), dimblock);

    swap_buffers();

    particles_vec1_.resize(matches_.nrows() * matches_.ncols());
    particles_vec2_.resize(matches_.nrows() * matches_.ncols());
    init_particles_vec<particle><<<dimgrid, dimblock>>>(thrust::raw_pointer_cast( &particles_vec1_[0]), *particles_);
    thrust::detail::normal_iterator<thrust::device_ptr<cuimg::i_short2> > end
      = thrust::remove_copy(particles_vec1_.begin(), particles_vec1_.end(), particles_vec2_.begin(), i_short2(-1, -1));
    particles_vec1_.swap(particles_vec2_);
       n_particles_ = end - particles_vec1_.begin();

    // n_particles_ = matches_.nrows() * matches_.ncols();
    //std::cout << "particles_vec_.size after " << particles_vec_.size() << std::endl;

    dim3 reduced_dimgrid(1 + n_particles_ / (dimblock.x * dimblock.y), dimblock.y, 1);

    i_short2 mvt = t_mvt.mvt();
    naive_matching_kernel2<typename F::kernel_type, particle><<<reduced_dimgrid, dimblock>>>
      //naive_matching_kernel2<typename F::kernel_type, particle><<<dimgrid, dimblock>>>
      (thrust::raw_pointer_cast( &particles_vec1_[0]), n_particles_, f, *particles_, *new_particles_, matches_,ls_matches, mvt, thrust::raw_pointer_cast( &compact_particles_[0])

#ifdef WITH_DISPLAY
,distance_, dg::widgets::Slider("age_filter").value()
#endif
                                                                                     );


    check_cuda_error();

    // check_robbers<particle><<<dimgrid, dimblock>>>(*particles_, *new_particles_, matches_, test_);
    filter_robbers<particle><<<reduced_dimgrid, dimblock>>>(thrust::raw_pointer_cast( &particles_vec1_[0]),
                                                            n_particles_,
                                                            *particles_, *new_particles_, matches_);
    filter_robbers<particle><<<reduced_dimgrid, dimblock>>>(thrust::raw_pointer_cast( &particles_vec1_[0]),
                                                            n_particles_,
                                                            *particles_, *new_particles_, matches_);
    check_cuda_error();

    fill(errors_, char(0));
    fill(fm_disp_, i_float4(0.f, 0.f, 0.f, 1.f));
    filter_false_matching<particle><<<reduced_dimgrid, dimblock>>>(thrust::raw_pointer_cast( &particles_vec1_[0]),
                                                                   n_particles_,
                                                                   *particles_, *new_particles_, matches_, errors_, fm_disp_);

    check_cuda_error();


    // check_robbers<particle><<<dimgrid, dimblock>>>(*particles_, *new_particles_, matches_, test2_);

    //if (!(frame_cpt % 5))
      create_particles_kernel<typename F::kernel_type, particle><<<dimgrid, dimblock>>>(f, *new_particles_, f.pertinence(), test_);

    check_cuda_error();
    return;


  }


  template <typename F>
  void
  naive_local_matcher_thrust<F>::display() const
  {
#ifdef WITH_DISPLAY
    image2d_f4 age(distance_.domain());
    dim3 dimblock(16, 16, 1);
    dim3 dimgrid = grid_dimension(distance_.domain(), dimblock);


    extract_age<particle><<<dimgrid, dimblock>>>
      (*new_particles_, age);
    dg::widgets::ImageView("distances") <<= dg::dl() - distance_ - test_ - test2_ + age - fm_disp_;
#endif
  }

  template <typename F>
  const device_image2d<typename naive_local_matcher_thrust<F>::particle>&
  naive_local_matcher_thrust<F>::particles() const
  {
    return *new_particles_;
  }



  template <typename F>
  const thrust::device_vector<typename naive_local_matcher_thrust<F>::particle>&
  naive_local_matcher_thrust<F>::compact_particles() const
  {
    return compact_particles_;
  }

  template <typename F>
  const image2d_s2&
  naive_local_matcher_thrust<F>::matches() const
  {
    return matches_;
  }

  template <typename F>
  const image2d_c&
  naive_local_matcher_thrust<F>::errors() const
  {
    return errors_;
  }

}

#endif // !  CUIMG_NAIVE_LOCAL_MATCHER_THRUST_HPP_
