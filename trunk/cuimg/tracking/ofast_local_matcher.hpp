#ifndef CUIMG_OFAST_LOCAL_MATCHER_HPP_
# define  CUIMG_OFAST_LOCAL_MATCHER_HPP_

#include <GL/glew.h>
#include <cuimg/gpu/cuda.h>

# ifndef NO_CUDA
#  include <thrust/remove.h>
# endif


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
# include <cuimg/gpu/device_image2d.h>
# include <cuimg/gpu/fill.h>
# include <cuimg/neighb2d_data.h>
# include <cuimg/neighb_iterator2d.h>
# include <cuimg/static_neighb2d.h>
# include <cuimg/point2d.h>
# include <cuimg/memset.h>

# include <cuimg/tracking/matcher_tools.h>

// extern unsigned it_counter;
// extern unsigned n_counter;

namespace cuimg
{

  // #define feat_tex UNIT_STATIC(feat_tex_id)

  // texture<float4, 2, cudaReadModeElementType> feat_tex;

#define init_particles_vec_sig(TG, T)                           \
  i_short2*, kernel_image2d<T>, &init_particles_vec<TG, T>

  template <unsigned target, typename T>
  __host__ __device__ void init_particles_vec(thread_info<target> ti,
                                              i_short2* particles_vec,
                                              kernel_image2d<T> particles)
  {
    point2d<int> p = thread_pos2d(ti);

    if (!particles.has(p)) return;

    i_short2 res(-1, -1);
    if (particles(p).age != 0)
      res = i_short2(p.row(), p.col());
    particles_vec[p.row() * particles.ncols() + p.col()] = res;
  }


#define naive_matching_kernel2_sig(TG, F, T)    \
  i_short2* ,                                   \
    unsigned ,                                  \
    F,                                          \
    kernel_image2d<T>,                          \
    kernel_image2d<T>,                          \
    kernel_image2d<i_short2> ,                  \
    kernel_image2d<F::feature_t> ,     \
    const kernel_image2d<i_short2> ,            \
    i_short2 ,                                  \
    T*,                                         \
    kernel_image2d<i_float4> ,                  \
    &naive_matching_kernel2<TG, F, T>


  template <unsigned target, typename F, typename T>
  __host__ __device__ void naive_matching_kernel2(thread_info<target> ti,
                                                  i_short2* particles_vec,
                                                  unsigned n_particles,
                                                  F f,
                                                  kernel_image2d<T> particles,
                                                  kernel_image2d<T> new_particles,
                                                  kernel_image2d<i_short2> matches,
                                                  kernel_image2d<typename F::feature_t> states,
                                                  const kernel_image2d<i_short2> ls_matches,
                                                  i_short2 mvt,
                                                  T* compact_particles
                                                  ,kernel_image2d<i_float4> dist
                                                  )
  {
    unsigned threadid = ti.blockIdx.x * ti.blockDim.x + ti.threadIdx.x;
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
    float p_pertinence = particles(p).pertinence;
    typename F::feature_t p_state = states(p);
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
      match_distance = f.distance_linear(p_state, p);

      //if (match_distance > 0.01f)
      if (false)
      {
        if (ls_p_found)
          prediction = ls_pred;
        else
          prediction = i_int2(p) + mvt;

        if (!f.s1().has(prediction)
            ||
            prediction.row() < 7 || prediction.row() >= particles.domain().nrows() - 7 ||
            prediction.col() < 7 || prediction.col() >= particles.domain().ncols() - 7
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

      if (!f.s1().has(prediction)
          ||
          prediction.row() < 7 || prediction.row() >= particles.domain().nrows() - 7 ||
          prediction.col() < 7 || prediction.col() >= particles.domain().ncols() - 7
          ) return;

      match = prediction;

      //match_distance = f.distance_linear(p_state, tex2D(feat_tex, prediction));
      match_distance = f.distance_linear(p_state, prediction);
    }

    //    if (match_distance > 0.01f)
    {

      unsigned match_i = 8;
      // n_counter++;
      for (int search = 0; search < 7; search++)
      {
        int i = c8_it[match_i][0];

        {
          point2d<int> n(prediction.row() + c8[i][0],
                         prediction.col() + c8[i][1]);
          {
            float d = f.distance_linear(p_state, n);
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
            float d = f.distance_linear(p_state, n);
            if (d < match_distance)
            {
              match = n;
              match_i = i;
              match_distance = d;
            }
          }
        }

        if (i_int2(prediction) == i_int2(match) ||
            match.row() < 7 || match.row() >= (particles.domain().nrows() - 7) ||
            match.col() < 7 || match.col() >= (particles.domain().ncols() - 7)
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

    if (f.pertinence()(match) < 0.001f) fault += 5;

    if (match_distance < (0.4f) && fault < 20)
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
        new_particles(match).speed = (new_speed * 3.f + particles(p).speed * 1.f) / 4.f;
      }
      //new_particles(match).state = f.current_frame()(match); //p_state;//


      // if (!(p_age % 50))
      //   new_particles(match).ipos = i_int2(p);
      // else
        new_particles(match).ipos = particles(p).ipos;

	if (norml2(new_speed) > 2)
	  states(match) =f.weighted_mean(p_state, 4.f,
					 match, 1.f);
	else
	  states(match) = p_state;

      // new_particles(match).state =
      //   f.weighted_mean(p_state, 1.f,
      //                   match, 4.f);

      for (unsigned i = 1; i < T::history_size; i++)
	new_particles(match).pos_history[i] = particles(p).pos_history[i - 1];
      new_particles(match).pos_history[0] = i_int2(match);

      new_particles(match).age = p_age + 1;
      new_particles(match).pertinence = f.pertinence()(match);
      new_particles(match).fault = fault;
      new_particles(match).pos = i_int2(match);

      particles(p) = new_particles(match);
      states(p) = states(match);

      compact_particles[threadid] = new_particles(match);

      matches(p) = i_short2(match.row(), match.col());

      // if (ls_p_found)
      //   dist(p) = i_float4(0.f, 0.7f, 0.f, 1.f);

    }
    else
    {
      compact_particles[threadid].pos = i_short2(-1, -1);
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
    // if (particles(p).age > age_filter)
    {
      dist(p) = i_float4(0.f, 0.f, 0.f, 0.f);
    }
    //dist(p) = p_state.o1;
#endif

  }

  template <typename F, unsigned HS>
  ofast_local_matcher<F, HS>::ofast_local_matcher(const domain_t& d)
    : t1_(d),
      t2_(d),
      distance_(d),
      test_(d),
      test2_(d),
      matches_(d),
      states_(d),
      errors_(d),
      fm_disp_(d),
      particles_vec1_(d.nrows() * d.ncols()),
      particles_vec2_(d.nrows() * d.ncols()),
      compact_particles_(d.nrows() * d.ncols()),
      n_particles_(0),
      frame_cpt(0)
  {
    particles_ = &t1_;
    new_particles_ = &t2_;

    particle p; p.age = 0;
    fill(*particles_, p);
    fill(*new_particles_, p);

    memset(*particles_, 0);
    memset(*new_particles_, 0);
    memset(errors_, 0);
  }

   template <typename F, unsigned HS>
  void
  ofast_local_matcher<F, HS>::swap_buffers()
  {
    std::swap(particles_, new_particles_);

    dim3 dimblock(128, 1, 1);
    dim3 dimgrid = grid_dimension(new_particles_->domain(), dimblock);

    memset(matches_, -1);
    memset(*new_particles_, 0);

  //  particle p; p.age = 0;
    // init_particles_kernel<particle><<<dimgrid, dimblock>>>(*new_particles_);
//    fill(*new_particles_, p);
  }


  template <typename F, unsigned HS>
  unsigned
  ofast_local_matcher<F, HS>::n_particles() const
  {
    return n_particles_;
  }

  template <typename F, unsigned HS>
  void
  ofast_local_matcher<F, HS>::update(F& f, global_mvt_thread<ofast_local_matcher<F, HS> >& t_mvt,
                                 const image2d_s2& ls_matches)
  {
    update(flag<F::target>(), f, t_mvt, ls_matches);
  }

#ifndef NO_CUDA
  template <typename F, unsigned HS>
  void
  ofast_local_matcher<F, HS>::update(const flag<GPU>&, F& f, global_mvt_thread<ofast_local_matcher<F, HS> >& t_mvt,
                                 const image2d_s2& ls_matches)
  {
    frame_cpt++;
    dim3 dimblock(128, 1, 1);
    dim3 dimgrid = grid_dimension(f.domain(), dimblock);

    swap_buffers();

    //bindTexture2d(*(device_image2d<float4>*)&(f.current_frame()), feat_tex);
    particles_vec1_.resize(matches_.nrows() * matches_.ncols());
    particles_vec2_.resize(matches_.nrows() * matches_.ncols());

    //std::cout << "particles_vec_.size " << particles_vec_.size() << std::endl;
    pw_call<init_particles_vec_sig(GPU, particle)>(flag<GPU>(), dimgrid, dimblock,
                                                   thrust::raw_pointer_cast( &particles_vec1_[0]), *particles_);

    // return;

    thrust::detail::normal_iterator<thrust::device_ptr<cuimg::i_short2> > end
      = thrust::remove_copy(particles_vec1_.begin(), particles_vec1_.end(), particles_vec2_.begin(), i_short2(-1, -1));
    particles_vec1_.swap(particles_vec2_);
    n_particles_ = end - particles_vec1_.begin();
    particles_vec1_.resize(n_particles_);

    // n_particles_ = matches_.nrows() * matches_.ncols();
    //std::cout << "particles_vec_.size after " << particles_vec_.size() << std::endl;


    if (n_particles_ > 0)
    {
      dim3 reduced_dimgrid(1 + n_particles_ / (dimblock.x * dimblock.y), dimblock.y, 1);

#ifdef WITH_DISPLAY
      distance_ = aggregate<float>::run(1.f, 0.f, 0.f, 1.f);
#endif
      i_short2 mvt = t_mvt.mvt() * 2;

      pw_call<naive_matching_kernel2_sig(GPU, typename F::kernel_type, particle)>
        (flag<GPU>(), reduced_dimgrid, dimblock,
         thrust::raw_pointer_cast( &particles_vec1_[0]),
         n_particles_, f, *particles_, *new_particles_, matches_, states_,
         ls_matches, mvt, thrust::raw_pointer_cast( &compact_particles_[0])
         ,distance_);

      pw_call<filter_robbers_sig(GPU, particle)>(flag<GPU>(), reduced_dimgrid, dimblock,
                                                 thrust::raw_pointer_cast( &particles_vec1_[0]),
                                                 n_particles_,
                                                 *particles_, *new_particles_, matches_);

      pw_call<filter_robbers_sig(GPU, particle)>(flag<GPU>(), reduced_dimgrid, dimblock,
                                                 thrust::raw_pointer_cast( &particles_vec1_[0]),
                                                 n_particles_,
                                                 *particles_, *new_particles_, matches_);
      check_cuda_error();


      memset(errors_, 0);
#ifdef WITH_STATS
      memset(errors_, 0);
      fill(fm_disp_, i_float4(0.f, 0.f, 0.f, 1.f));
#else
# ifdef WITH_DISPLAY
      fill(fm_disp_, i_float4(0.f, 0.f, 0.f, 1.f));
# endif
#endif

      pw_call<filter_false_matching_sig(GPU, particle)>(flag<GPU>(),reduced_dimgrid, dimblock,
                                                        thrust::raw_pointer_cast( &particles_vec1_[0]),
                                                        n_particles_,
                                                        *particles_, *new_particles_,
                                                        matches_, errors_, fm_disp_);


    }
    check_cuda_error();

    check_cuda_error();


    // check_robbers<particle><<<dimgrid, dimblock>>>(*particles_, *new_particles_, matches_, test2_);

    if (!(frame_cpt % 5))
      pw_call<create_particles_kernel_sig(GPU, typename F::kernel_type, particle)>(flag<GPU>(), dimgrid, dimblock, f, *new_particles_, f.pertinence(), states_);


    check_cuda_error();
    return;


  }

#endif // ! NO_CUDA

  template <typename F, unsigned HS>
  void
  ofast_local_matcher<F, HS>::update(const flag<CPU>&, F& f, global_mvt_thread<ofast_local_matcher<F, HS> >& t_mvt,
                                 const image2d_s2& ls_matches)
  {
    frame_cpt++;
    dim3 dimblock(f.domain().ncols(), 1, 1);

    dim3 dimgrid = grid_dimension(f.domain(), dimblock);

    swap_buffers();

    // particles_vec1_.clear();

    if (n_particles_ > 0)
    {
      // dim3 r_dimblock(n_particles_ / 8, 1, 1);
      dim3 reduced_dimgrid(1 + n_particles_ / (dimblock.x * dimblock.y), 1, 1);
      // dim3 reduced_dimgrid(9, 1, 1);

#ifdef WITH_DISPLAY
      distance_ = aggregate<float>::run(1.f, 0.f, 0.f, 1.f);
#endif
      i_short2 mvt = t_mvt.mvt() * 2;
      pw_call<naive_matching_kernel2_sig(CPU, typename F::kernel_type, particle)>
        (flag<CPU>(), reduced_dimgrid, dimblock,
         &particles_vec1_[0],
         n_particles_, f, *particles_, *new_particles_, matches_, states_,
         ls_matches, mvt, &compact_particles_[0]
         ,distance_);

      pw_call<filter_robbers_sig(CPU, particle)>(flag<CPU>(), reduced_dimgrid, dimblock,
                                                 &particles_vec1_[0],
                                                 n_particles_,
                                                 *particles_, *new_particles_, matches_);

      pw_call<filter_robbers_sig(CPU, particle)>(flag<CPU>(), reduced_dimgrid, dimblock,
                                                 &particles_vec1_[0],
                                                 n_particles_,
                                                 *particles_, *new_particles_, matches_);
      check_cuda_error();

     ImageView("frame2") << distance_
                         << dg::widgets::show;


      memset(errors_, 0);
#ifdef WITH_STATS
      memset(errors_, 0);
      fill(fm_disp_, i_float4(0.f, 0.f, 0.f, 1.f));
#else
# ifdef WITH_DISPLAY
      fill(fm_disp_, i_float4(0.f, 0.f, 0.f, 1.f));
# endif
#endif

      // if (!(frame_cpt % 30))
      pw_call<filter_false_matching_sig(CPU, particle)>(flag<CPU>(), reduced_dimgrid, dimblock,
                                                        &particles_vec1_[0],
                                                        n_particles_,
                                                        *particles_, *new_particles_,
                                                        matches_, errors_, fm_disp_);


    }

    check_cuda_error();
    check_cuda_error();


    // check_robbers<particle><<<dimgrid, dimblock>>>(*particles_, *new_particles_, matches_, test2_);

    //if (!(frame_cpt % 5))
      pw_call<create_particles_kernel_sig(CPU, typename F::kernel_type, particle)>(flag<CPU>(), dimgrid, dimblock, typename F::kernel_type(f), *new_particles_, f.pertinence(), states_);

    particles_vec1_.resize(matches_.nrows() * matches_.ncols());
    n_particles_ = 0;
    for (unsigned r = 0; r < particles_->nrows(); r++)
      for (unsigned c = 0; c < particles_->ncols(); c++)
        if ((*new_particles_)(r, c).age != 0)
        {
          particles_vec1_[n_particles_] = i_short2(r, c);
	  compact_particles_[n_particles_] = (*new_particles_)(r,c);
          n_particles_++;
        }

    particles_vec1_.resize(n_particles_);
    compact_particles_.resize(n_particles_);
    check_cuda_error();
    return;


  }

  template <typename F, unsigned HS>
  void
  ofast_local_matcher<F, HS>::display() const
  {
#ifdef WITH_DISPLAY
    image2d_f4 age(distance_.domain());
    dim3 dimblock(16, 16, 1);
    dim3 dimgrid = grid_dimension(distance_.domain(), dimblock);


    // extract_age<particle><<<dimgrid, dimblock>>>
    //   (*new_particles_, age);
    // dg::widgets::ImageView("distances") <<= dg::dl() - distance_ - test_ - test2_ + age - fm_disp_;
#endif
  }

  template <typename F, unsigned HS>
  const typename ofast_local_matcher<F, HS>::image2d_P&
  ofast_local_matcher<F, HS>::particles() const
  {
    return *new_particles_;
  }

  template <typename F, unsigned HS>
  typename ofast_local_matcher<F, HS>::particle_vector&
  ofast_local_matcher<F, HS>::compact_particles()
  {
    return compact_particles_;
  }

  template <typename F, unsigned HS>
  const typename ofast_local_matcher<F, HS>::particle_vector&
  ofast_local_matcher<F, HS>::compact_particles() const
  {
    return compact_particles_;
  }

  template <typename F, unsigned HS>
  const typename ofast_local_matcher<F, HS>::image2d_s2&
  ofast_local_matcher<F, HS>::matches() const
  {
    return matches_;
  }

  template <typename F, unsigned HS>
  const typename ofast_local_matcher<F, HS>::image2d_c&
  ofast_local_matcher<F, HS>::errors() const
  {
    return errors_;
  }

}

#endif // !  CUIMG_OFAST_LOCAL_MATCHER_HPP_
