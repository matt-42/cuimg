#ifndef CUIMG_MULTI_SCALE_TRACKER_HPP_
# define CUIMG_MULTI_SCALE_TRACKER_HPP_

# include <cuimg/improved_builtin.h>
# include <cuimg/gpu/tracking/multi_scale_tracker.h>
# include <cuimg/gpu/mipmap.h>

namespace cuimg
{

  template <typename F, template <class>  class SA_>
  inline
  multi_scale_tracker<F, SA_>::multi_scale_tracker(const D& d)
    : frame_uc3_(d),
      frame_(d),
      traj_tracer_(d),
      dummy_matches_(d),
      mvt_detector_thread_(d)
  {
    pyramid_ = allocate_mipmap(i_float1(), frame_, PS);
    pyramid_tmp1_ = allocate_mipmap(i_float1(), frame_, PS);
    pyramid_tmp2_ = allocate_mipmap(i_float1(), frame_, PS);

    pyramid_display1_ = allocate_mipmap(i_float4(), frame_, PS);
    pyramid_display2_ = allocate_mipmap(i_float4(), frame_, PS);
    pyramid_speed_ = allocate_mipmap(i_float4(), frame_, PS);
    for (unsigned i = 0; i < pyramid_.size(); i++)
    {
      feature_.push_back(new F(pyramid_[i].domain()));
      matcher_.push_back(new SA(pyramid_[i].domain()));
    }

    fill(dummy_matches_, i_short2(-1, -1));
  }

  template <typename F, template <class>  class SA_>
  multi_scale_tracker<F, SA_>::~multi_scale_tracker()
  {
    for (unsigned i = 0; i < pyramid_.size(); i++)
    {
      delete feature_[i];
      delete matcher_[i];
    }
  }

  template <typename F, template <class>  class SA_>
  inline
  const multi_scale_tracker<F, SA_>::D&
  multi_scale_tracker<F, SA_>::domain() const
  {
    return frame_.domain();
  }

  template <typename P>
  __global__ void draw_particles(kernel_image2d<i_float1> frame,
                                 kernel_image2d<P> pts,
                                 kernel_image2d<i_float4> out,
                                 int age_filter)
  {
    point2d<int> p = thread_pos2d();
    if (!frame.has(p))
      return;

    //if (pts(p).age < age_filter) out(p) = frame(p);
    if (pts(p).age < age_filter) out(p) = i_float4(frame(p).x, frame(p).x, frame(p).x, 1.f);
    else out(p) = i_float4(1.f, 0, 0, 1.f);
  }

  template <typename P>
  __global__ void draw_speed(kernel_image2d<P> track, kernel_image2d<i_float4> out)
  {
    point2d<int> p = thread_pos2d();
    if (!track.has(p))
      return;

    if (!track(p).age)
    {
       out(p) = i_float4(0, 0, 0, 1.f);
       return;
    }

    i_float2 speed(track(p).speed);
    float module = norml2(speed);

    float R = ::abs(2 * track(p).speed.x + module) / 14.f;
    float G = ::abs(-track(p).speed.x - track(p).speed.y + module) / 14.f;
    float B = ::abs(2 * track(p).speed.y + module) / 14.f;

    R = R < 0.f ? 0.f : R;
    G = G < 0.f ? 0.f : G;
    B = B < 0.f ? 0.f : B;
    out(p) = i_float4(R, G, B, 1.f);

    // out(p) = i_float4(diff, diff, diff, 1.f);
  }


  template <typename P>
  __global__ void of_reconstruction(kernel_image2d<i_float4> speed, kernel_image2d<i_float1> in, kernel_image2d<i_float4> out)
  {
    point2d<int> p = thread_pos2d();
    if (!speed.has(p))
      return;

    // if (speed(p) != i_float4(0, 0, 0, 1.f))
    // {
    //   out(p) = speed(p);
    //   return;
    // }

    i_float4 res = zero();
    int cpt = 0;
    for_all_in_static_neighb2d(p, n, c49) if (speed.has(n) && speed(n) != i_float4(0, 0, 0, 1.f))
    {
       if (norml2(in(p) - in(n)) < (30.f / 256.f))
       {
          res += speed(n);
          cpt++;
       }
    }

    if (cpt >= 2)
      out(p) = res / cpt;
    else
      out(p) = i_float4(0, 0, 0, 1.f);
  }

  template <typename P>
  __global__ void sub_global_mvt(kernel_image2d<P> particles, i_short2 mvt)
  {
    point2d<int> p = thread_pos2d();
    if (!particles.has(p) || particles(p).age == 0)
      return;

    if (particles(p).age > 2)
    {
      //particles(p).acceleration -= mvt;
      particles(p).speed -= mvt * 3.f / 4.f;
    }
    else
      particles(p).speed -= mvt;
  }


    template <typename P>
  __global__ void draw_particles(kernel_image2d<P> pts,
                                 kernel_image2d<i_float4> out,
                                 int age_filter)
  {
    point2d<int> p = thread_pos2d();
    if (!out.has(p))
      return;

    if (pts(p).age < age_filter) out(p) = i_float4(0.f, 0.f, 0.f, 1.f);
    else out(p) = i_float4(1.f, 0.f, 0, 1.f);
  }

  template <typename P>
  __global__ void draw_particle_pertinence(kernel_image2d<i_float1> pertinence,
                                           kernel_image2d<P> pts,
                                           kernel_image2d<i_float4> out,
                                           int age_filter)
  {
    point2d<int> p = thread_pos2d();
    if (!out.has(p))
      return;

    if (pts(p).age > age_filter) out(p) = i_float4(1.f, 0.f, 0.f, 1.f);
    else
      out(p) = i_float4(pertinence(p), pertinence(p), pertinence(p), 1.f);
  }

  template <typename F, template <class>  class SA_>
  inline
  void multi_scale_tracker<F, SA_>::update(const host_image2d<i_uchar3>& in)
  {
    copy(in, frame_uc3_);
    frame_ = (get_x(frame_uc3_) + get_y(frame_uc3_) + get_z(frame_uc3_)) / (255 * 3.f);


    update_mipmap(frame_, pyramid_, pyramid_tmp1_, pyramid_tmp2_, PS);

    i_short2 mvt(0, 0);

  /*  ImageView("frame test") <<= dg::dl() - pyramid_[4] - pyramid_[3] - pyramid_[2] - pyramid_[1] - pyramid_[0] +
                                           pyramid_tmp1_[4] - pyramid_tmp1_[3] - pyramid_tmp1_[2] - pyramid_tmp1_[1] - pyramid_tmp1_[0] +
                                           pyramid_tmp2_[4] - pyramid_tmp2_[3] - pyramid_tmp2_[2] - pyramid_tmp2_[1] - pyramid_tmp2_[0];


    */

    for (int l = pyramid_.size() - 1; l >= 0; l--)
    {
      mvt *= 2;
      feature_[l]->update(pyramid_[l]);
      if (l != pyramid_.size() - 1)
        matcher_[l]->update(*(feature_[l]), mvt_detector_thread_, matcher_[l+1]->matches());
      else
        matcher_[l]->update(*(feature_[l]), mvt_detector_thread_, dummy_matches_);

      if (l > 1)
      {
        mvt_detector_thread_.update(*(matcher_[l]), l);
      }

      dim3 dimblock(16, 16);
      dim3 dimgrid = dim3(grid_dimension(matcher_[l]->particles().domain(),
                                         dimblock));

      sub_global_mvt<P><<<dimgrid, dimblock>>>
        (matcher_[l]->particles(), mvt);

      //if (l == pyramid_.size() - 3) mvt_detector.display();
    }


#ifdef WITH_DISPLAY

//#define TD 0
    unsigned TD = Slider("display_scale").value();
    if (TD >= PS) TD = PS - 1;
    traj_tracer_.update(matcher_[TD]->matches(), matcher_[TD]->particles());

    feature_[TD]->display();
    matcher_[TD]->display();
    mvt_detector_thread_.mvt_detector().display();
    traj_tracer_.display("trajectories", pyramid_[TD]);

    dim3 dimblock(16, 16);
    dim3 dimgrid = dim3(grid_dimension(matcher_[TD]->particles().domain(), dimblock));
    draw_particles<P><<<dimgrid, dimblock>>>(pyramid_[TD], matcher_[TD]->particles(), pyramid_display1_[TD], Slider("age_filter").value());
    draw_particles<P><<<dimgrid, dimblock>>>(matcher_[TD]->particles(), pyramid_display2_[TD], Slider("age_filter").value());

    image2d<i_float4> particle_pertinence(matcher_[TD]->particles().domain());
    image2d<i_float4> of(matcher_[TD]->particles().domain());
    draw_particle_pertinence<P><<<dimgrid, dimblock>>>(feature_[TD]->pertinence(), matcher_[TD]->particles(),
                                                       particle_pertinence, Slider("age_filter").value());
    draw_speed<P><<<dimgrid, dimblock>>>(matcher_[TD]->particles(), pyramid_speed_[TD]);
    of_reconstruction<P><<<dimgrid, dimblock>>>(pyramid_speed_[TD], pyramid_[TD], of);
    ImageView("frame") <<= dg::dl() - pyramid_display1_[TD] - pyramid_display2_[TD] - pyramid_speed_[TD] + of - particle_pertinence;
#endif
  }

}

#endif // !CUIMG_MULTI_SCALE_TRACKER_HPP_

