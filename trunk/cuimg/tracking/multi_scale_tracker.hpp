#ifndef CUIMG_MULTI_SCALE_TRACKER_HPP_
# define CUIMG_MULTI_SCALE_TRACKER_HPP_

# include <cuimg/improved_builtin.h>
# include <cuimg/tracking/multi_scale_tracker.h>
# include <cuimg/gpu/mipmap.h>
# include <cuimg/dsl/dsl_cast.h>
# include <cuimg/pw_call.h>
# include <dige/recorder.h>

namespace cuimg
{

  template <typename F, typename SA>
  inline
  multi_scale_tracker<F, SA>::multi_scale_tracker(const D& d)
    : frame_uc3_(d),
      frame_(d),
      accelerations_(PS),
      accelerations_(PS),
      dummy_matches_(d),
      mvt_detector_thread_(d)
  {
    pyramid_ = allocate_mipmap(V(), frame_, PS);
    pyramid_tmp1_ = allocate_mipmap(V(), frame_, PS);
    pyramid_tmp2_ = allocate_mipmap(V(), frame_, PS);

    pyramid_display1_ = allocate_mipmap(i_float4(), frame_, PS);
    pyramid_display2_ = allocate_mipmap(i_float4(), frame_, PS);
    pyramid_speed_ = allocate_mipmap(i_float4(), frame_, PS);
    for (unsigned i = 0; i < pyramid_.size(); i++)
    {
      feature_.push_back(new F(pyramid_[i].domain()));
      matcher_.push_back(new SA(pyramid_[i].domain()));
      accelerations_[i] = i_short2(0, 0);
    }

    fill(dummy_matches_, i_short2(-1, -1));

#ifdef WITH_DISPLAY
    //  ImageView("video", d.ncols() * 2, d.nrows() * 2);
#endif

  }

  template <typename F, typename SA>
  multi_scale_tracker<F, SA>::~multi_scale_tracker()
  {
    for (unsigned i = 0; i < pyramid_.size(); i++)
    {
      delete feature_[i];
      delete matcher_[i];
    }
  }

  template <typename F, typename SA>
  inline
  const typename multi_scale_tracker<F, SA>::D&
  multi_scale_tracker<F, SA>::domain() const
  {
    return frame_.domain();
  }

  template <typename V>
  __host__ __device__ void plot_c4(kernel_image2d<V>& out, point2d<int>& p, const V& value)
  {
    for_all_in_static_neighb2d(p, n, c5) if (out.has(n))
      out(n) = value;
    out(p) = i_float4(0.f, 0.f, 0.f, 1.f);
  }

  template <unsigned target, typename P>
  __host__ __device__ void draw_particles(thread_info<target> ti,
                                          kernel_image2d<i_float1> frame,
                                          kernel_image2d<P> pts,
                                          kernel_image2d<i_float4> out,
                                          int age_filter)
  {
    point2d<int> p = thread_pos2d(ti);
    if (!frame.has(p))
      return;

    //if (pts(p).age < age_filter) out(p) = frame(p);
    if (pts(p).age >= age_filter) //else out(p) = i_float4(1.f, 0, 0, 1.f);
      plot_c4(out, p, i_float4(1.f, 0.f, 0, 1.f));
  }

#define draw_particles_sig(T, P) kernel_image2d<i_float1>, kernel_image2d<P>, \
    kernel_image2d<i_float4>, int, &draw_particles<T, P>

  template <unsigned target, typename P>
  __host__ __device__ void draw_speed(thread_info<target> ti, kernel_image2d<P> track, kernel_image2d<i_float4> out)
  {
    point2d<int> p = thread_pos2d(ti);
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
  }

#define draw_speed_sig(T, P) kernel_image2d<P>, kernel_image2d<i_float4>, &draw_speed<T, P>

  template <unsigned target, typename P>
  __host__ __device__ void of_reconstruction(thread_info<target> ti, kernel_image2d<i_float4> speed, kernel_image2d<i_float1> in, kernel_image2d<i_float4> out)
  {
    point2d<int> p = thread_pos2d(ti);
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

#define of_reconstruction_sig(T, P) kernel_image2d<i_float4>, kernel_image2d<i_float1>, \
            kernel_image2d<i_float4>, &of_reconstruction<T, P>

  template <unsigned target, typename P>
  __host__ __device__ void sub_global_mvt(thread_info<target> ti, const i_short2* particles_vec,
                                          unsigned n_particles, kernel_image2d<P> particles, i_short2 mvt)
  {
    unsigned threadid = ti.blockIdx.x * ti.blockDim.x + ti.threadIdx.x;
    if (threadid >= n_particles) return;
    point2d<int> p = particles_vec[threadid];

    // point2d<int> p = thread_pos2d();
    if (!particles.has(p))
      return;

    i_float2 speed = particles(p).speed;
    if (particles(p).age > 2)
    {
      //particles(p).acceleration -= mvt;
      speed -= mvt * 3.f / 4.f;
    }
    else
      speed -= mvt;

    particles(p).speed = speed;
  }

#define sub_global_mvt_sig(T, P) const i_short2*, unsigned,      \
    kernel_image2d<P>, i_short2, &sub_global_mvt<T, P>

  template <unsigned target, typename P>
  __host__ __device__ void draw_particles2(thread_info<target> ti, kernel_image2d<P> pts,
                                          kernel_image2d<i_float4> out,
                                          int age_filter)
  {
    point2d<int> p = thread_pos2d(ti);
    if (!out.has(p))
      return;

    if (pts(p).age < age_filter) out(p) = i_float4(0.f, 0.f, 0.f, 1.f);
    // else plot_c4(out, p, i_float4(1.f, 0, 0, 1.f));
    else out(p) = i_float4(1.f, 0.f, 0, 1.f);
  }

#define draw_particles2_sig(T, P) kernel_image2d<P>, kernel_image2d<i_float4>, int, &draw_particles2<T, P>

  template <unsigned target, typename P>
  __host__ __device__ void draw_particle_pertinence(thread_info<target> ti,
                                                    kernel_image2d<i_float1> pertinence,
                                                    kernel_image2d<P> pts,
                                                    kernel_image2d<i_float4> out,
                                                    int age_filter)
  {
    point2d<int> p = thread_pos2d(ti);
    if (!out.has(p))
      return;

    if (pts(p).age > age_filter) plot_c4(out, p, i_float4(1.f, 0, 0, 1.f));//out(p) = i_float4(1.f, 0.f, 0.f, 1.f);
    // else
    //   out(p) = i_float4(pertinence(p), pertinence(p), pertinence(p), 1.f);
  }

#define draw_particle_pertinence_sig(T, P) kernel_image2d<i_float1>, kernel_image2d<P>, \
    kernel_image2d<i_float4>, int, &draw_particle_pertinence<T, P>


extern "C" {
  __inline__ uint64_t rdtsc(void) {
    uint32_t lo, hi;
    __asm__ __volatile__ (      // serialize
    "xorl %%eax,%%eax \n        cpuid"
    ::: "%rax", "%rbx", "%rcx", "%rdx");
    /* We cannot use "=A", since this would use %rax on x86_64 and return only the lower 32bits of the TSC */
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return (uint64_t)hi << 32 | lo;
  }
}

#define BENCH_CPP_START(M)                      \
  unsigned long long cycles_##M = rdtsc();

#define BENCH_CPP_END(M, NP)                                            \
  cycles_##M = rdtsc() - cycles_##M;                                    \
  typedef unsigned long long ull;                                       \
  std::cout << "[" << #M << "]\tCycles per pixel: " << std::fixed << float((double(cycles_##M) / (NP))) << std::endl;

  template <typename I, typename O>
  void to_graylevelf(const Image2d<I>& in_, Image2d<O>& out_)
  {
    const I& in = exact(in_);
    O& out = exact(out_);

#pragma omp parallel for schedule(static, 8)
    for (unsigned r = 0; r < in.nrows(); r++)
      for (unsigned c = 0; c < in.ncols(); c++)
      {
        i_uchar3 x = in(r, c);
        out(r, c) = (int(x.x) + x.y + x.z) / (3 * 255.f);
      }
  }

  template <typename I, typename O>
  void to_graylevel(const Image2d<I>& in_, Image2d<O>& out_, gl01f)
  {
    const I& in = exact(in_);
    O& out = exact(out_);
    out = (get_x(in) + get_y(in) + get_z(in)) / (3.f * 255.f);
  }

  template <typename I, typename O>
  void to_graylevel(const Image2d<I>& in_, Image2d<O>& out_, gl8u)
  {
    const I& in = exact(in_);
    O& out = exact(out_);
    out = dsl_cast<unsigned char>::run(((get_x(in)) + get_y(in) + get_z(in)) / (3));
  }

  template <typename I, typename J>
  void prepare_input_frame(const flag<CPU>&, const host_image2d<i_uchar3>& in, I& frame, J&)
  {
    to_graylevel(in, frame, typename I::value_type());
    // ImageView("video", 400, 400) << frame << dg::widgets::show;
  }

  template <typename I, typename J>
  void prepare_input_frame(const flag<GPU>&, const host_image2d<i_uchar3>& in, I& frame, J& tmp_uc3)
  {
    copy(in, tmp_uc3);
    to_graylevel(tmp_uc3, frame, typename I::value_type());
  }

  template <typename F, typename SA>
  inline
  void multi_scale_tracker<F, SA>::update(const host_image2d<i_uchar3>& in)
  {
    prepare_input_frame(flag<target>(), in, frame_, frame_uc3_);

    if (target == unsigned(CPU))
      update_mipmap(frame_, pyramid_, pyramid_tmp1_, pyramid_tmp2_, PS, 0, dim3(in.ncols(), 2));
    else
      update_mipmap(frame_, pyramid_, pyramid_tmp1_, pyramid_tmp2_, PS, 0, dim3(16, 16));

    mvt_detector_thread_.reset_mvt();
    for (int l = pyramid_.size() - 2; l >= 0; l--)
    {
      feature_[l]->update(pyramid_[l], pyramid_[l + 1]);
      if (l != pyramid_.size() - 1)
        matcher_[l]->update(*(feature_[l]), mvt_detector_thread_, matcher_[l+1]->matches());
      else
        matcher_[l]->update(*(feature_[l]), mvt_detector_thread_, dummy_matches_);

      accelerations_[l] = mvt_detector_thread_.mvt() * 2;

      if (l >= 0)
      {
        mvt_detector_thread_.update(*(matcher_[l]), l);
      }

      dim3 dimblock(128, 1);
      dim3 reduced_dimgrid(1 + matcher_[l]->n_particles() / (dimblock.x * dimblock.y), dimblock.y, 1);

#ifndef NO_CUDA
      pw_call<sub_global_mvt_sig(target, P)>(flag<target>(), reduced_dimgrid, dimblock,
					     thrust::raw_pointer_cast(&matcher_[l]->particle_positions()[0]),
					     matcher_[l]->n_particles(), matcher_[l]->particles(),
					     mvt_detector_thread_.mvt());
#else
      pw_call<sub_global_mvt_sig(target, P)>(flag<target>(), reduced_dimgrid, dimblock,
					     &(matcher_[l]->particle_positions()[0]),
					     matcher_[l]->n_particles(), matcher_[l]->particles(),
					     mvt_detector_thread_.mvt());
#endif

      // if (l == pyramid_.size() - 3) mvt_detector_thread_.display();
    }

#ifdef WITH_DISPLAY
    display();
#endif

  }


  template <typename F, typename SA>
  inline
  void multi_scale_tracker<F, SA>::display()
  {
#ifdef WITH_DISPLAY

    flag<target> f;

    //#define TD 0
    unsigned TD = Slider("display_scale").value();
    if (TD >= PS) TD = PS - 1;

    D d = matcher_[TD]->particles().domain();
    dim3 db(16, 16);
    dim3 dg = dim3(grid_dimension(d, db));

    feature_[TD]->display();
    matcher_[TD]->display();
    mvt_detector_thread_.mvt_detector().display();

    pyramid_display1_[TD] = aggregate<float>::run(get_x(pyramid_[TD]), get_x(pyramid_[TD]), get_x(pyramid_[TD]), 1.f);

    pw_call<draw_particles_sig(target, P)>(f,dg,db,pyramid_[TD], matcher_[TD]->particles(), pyramid_display1_[TD], Slider("age_filter").value());

    pw_call<draw_particles2_sig(target, P)>(f, dg,db, matcher_[TD]->particles(), pyramid_display2_[TD], Slider("age_filter").value());

    image2d_f4 particle_pertinence(matcher_[TD]->particles().domain());
    image2d_f4 of(matcher_[TD]->particles().domain());

    particle_pertinence = aggregate<float>::run(get_x(feature_[TD]->pertinence()),
                                                get_x(feature_[TD]->pertinence()),
                                                get_x(feature_[TD]->pertinence()), 1.f);

    // pw_call<draw_particle_pertinence_sig(target, P)>(f,dg,db,feature_[TD]->pertinence(), matcher_[TD]->particles(),
    //                                          particle_pertinence, Slider("age_filter").value());

    pw_call<draw_speed_sig(target, P)>
      (f, dg,db, kernel_image2d<P>(matcher_[TD]->particles()),
       kernel_image2d<i_float4>(pyramid_speed_[TD]));

    pw_call<of_reconstruction_sig(target, P)>(f,dg,db,pyramid_speed_[TD], pyramid_[TD], of);

    // ImageView("frame") <<= dg::dl() - frame_ - pyramid_display1_[TD] - pyramid_display2_[TD] - pyramid_speed_[TD] + of - particle_pertinence;

    ImageView("frame") <<= dg::dl() - pyramid_display1_[TD] - particle_pertinence;

    // dg::record("views.avi") <<= ImageView("video") <<= dg::dl() - pyramid_display1_[TD] - feature_[TD]->pertinence() +
    //   traj_tracer_.straj() - of;
#endif
  }

  template <typename F, typename SA>
  inline
  typename multi_scale_tracker<F, SA>::particle_vector&
  multi_scale_tracker<F, SA>::particles(unsigned scale)
  {
    if (scale < PS)
      return matcher_[scale]->compact_particles();
    else
      return matcher_[PS - 1]->compact_particles();
  }

  template <typename F, typename SA>
  inline
  const typename multi_scale_tracker<F, SA>::image2d_P&
  multi_scale_tracker<F, SA>::particles_img(unsigned scale) const
  {
    if (scale < PS)
      return matcher_[scale]->particles();
    else
      return matcher_[PS - 1]->particles();
  }

  template <typename F, typename SA>
  inline
  unsigned
  multi_scale_tracker<F, SA>::nparticles(unsigned scale) const
  {
    if (scale < PS)
      return matcher_[scale]->n_particles();
    else
      return matcher_[PS - 1]->n_particles();
  }

  template <typename F, typename SA>
  inline
  const typename multi_scale_tracker<F, SA>::image2d_s2&
  multi_scale_tracker<F, SA>::matches(unsigned scale) const
  {
    if (scale < PS)
      return matcher_[scale]->matches();
    else
      return matcher_[PS - 1]->matches();
  }

  template <typename F, typename SA>
  inline
  const typename multi_scale_tracker<F, SA>::image2d_c&
  multi_scale_tracker<F, SA>::errors() const
  {
    return matcher_[0]->errors();
  }

  template <typename F, typename SA>
  inline
  i_short2
  multi_scale_tracker<F, SA>::camera_acceleration(unsigned scale) const
  {
    return accelerations_[scale];
  }

}

#endif // !CUIMG_MULTI_SCALE_TRACKER_HPP_

