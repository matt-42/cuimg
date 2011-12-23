#ifndef CUIMG_MULTI_SCALE_TRACKER_HPP_
# define CUIMG_MULTI_SCALE_TRACKER_HPP_

# include <cuimg/gpu/tracking/multi_scale_tracker.h>
# include <cuimg/gpu/mipmap.h>

namespace cuimg
{

  inline
  multi_scale_tracker::multi_scale_tracker(const D& d)
    : frame_uc3_(d),
      frame_(d),
      p_display1_(D(d.nrows() / 2, d.ncols() / 2)),
      //p_display1_(D(d.nrows() / 1, d.ncols() / 1)),
      p_display2_(p_display1_.domain()),
      traj_tracer_(p_display1_.domain())
  {
    pyramid_ = allocate_mipmap(i_float1(), frame_, PS);
    pyramid_tmp1_ = allocate_mipmap(i_float1(), frame_, PS);
    pyramid_tmp2_ = allocate_mipmap(i_float1(), frame_, PS);

    for (unsigned i = 0; i < pyramid_.size(); i++)
    {
      feature_.push_back(new F(pyramid_[i].domain()));
      matcher_.push_back(new SA(pyramid_[i].domain()));
      matches_.push_back(host_image2d<i_short2>(pyramid_[i].domain()));
    }
  }

  multi_scale_tracker::~multi_scale_tracker()
  {
    for (unsigned i = 0; i < pyramid_.size(); i++)
    {
      delete feature_[i];
      delete matcher_[i];
    }
  }

  inline
  const multi_scale_tracker::D&
  multi_scale_tracker::domain() const
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

  inline
  void multi_scale_tracker::update(const host_image2d<i_uchar3>& in)
  {
    copy(in, frame_uc3_);
    frame_ = (get_x(frame_uc3_) + get_y(frame_uc3_) + get_z(frame_uc3_)) / (255 * 3.f);


    update_mipmap(frame_, pyramid_, pyramid_tmp1_, pyramid_tmp2_, PS);

    i_short2 mvt(0, 0);

  /*  ImageView("frame test") <<= dg::dl() - pyramid_[4] - pyramid_[3] - pyramid_[2] - pyramid_[1] - pyramid_[0] +
                                           pyramid_tmp1_[4] - pyramid_tmp1_[3] - pyramid_tmp1_[2] - pyramid_tmp1_[1] - pyramid_tmp1_[0] +
                                           pyramid_tmp2_[4] - pyramid_tmp2_[3] - pyramid_tmp2_[2] - pyramid_tmp2_[1] - pyramid_tmp2_[0];


    */
    for (int l = pyramid_.size() - 1; l >= 1; l--)
    {
      mvt *= 2;
      feature_[l]->update(pyramid_[l]);
      matcher_[l]->update(*(feature_[l]), mvt);
      copy(matcher_[l]->matches(), matches_[l]);
      mvt = mvt_detector.estimate(matches_[l]);
      //if (l == pyramid_.size() - 3) mvt_detector.display();
    }


    mvt *= 2;
    feature_[0]->update(pyramid_[0]);
    matcher_[0]->update(*(feature_[0]), mvt);

    traj_tracer_.update(matcher_[1]->matches(), matcher_[1]->particles());

#ifdef WITH_DISPLAY
    feature_[1]->display();
    matcher_[1]->display();
    mvt_detector.display();
    traj_tracer_.display("trajectories", pyramid_[1]);

    dim3 dimblock(16, 16);
    dim3 dimgrid = dim3(grid_dimension(matcher_[1]->particles().domain(), dimblock));
    draw_particles<P><<<dimgrid, dimblock>>>(pyramid_[1], matcher_[1]->particles(), p_display1_, Slider("age_filter").value());
    draw_particles<P><<<dimgrid, dimblock>>>(matcher_[1]->particles(), p_display2_, Slider("age_filter").value());
    ImageView("frame") <<= dg::dl() - p_display1_ - p_display2_;
#endif
  }

}

#endif // !CUIMG_MULTI_SCALE_TRACKER_HPP_
