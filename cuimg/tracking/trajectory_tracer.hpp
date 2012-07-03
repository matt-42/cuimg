
#include <cuimg/gpu/cuda.h>


# include <cuimg/dige.h>
# include <cuimg/dsl/threshold.h>
# include <cuimg/dsl/norml2.h>
# include <cuimg/gpu/fill.h>


#include <cuimg/static_neighb2d.h>
#include <cuimg/neighb_iterator2d.h>

# include <dige/widgets/image_view.h>
# include <dige/image.h>

# include <cuimg/tracking/trajectory_tracer.h>

using namespace dg::widgets;
using dg::dl;

#ifdef NVCC

template <target TG>
inline
trajectory_tracer<TG>::trajectory_tracer(const domain_t& d)
: rand_colors_(d),
  display_image_(d),
  traj_heads_(d),
  traj_(d),
  straj_(d),
  traces1_(d),
  traces2_(d),
  age_(d)
{
  traj_ = aggregate<float>::run(0.f, 0.f, 0.f, 1.f);
  age_ = aggregate<short>::run(0);

  trace t;
  t.reset();

  fill(traces1_, t);
  fill(traces2_, t);

  // cudaMemset2D(traces1_.data(), traces1_.pitch(), 0,
  //              traces1_.pitch(), traces1_.nrows());
  // cudaMemset2D(traces2_.data(), traces2_.pitch(), 0,
  //              traces2_.pitch(), traces2_.nrows());

#ifndef NO_CUDA
  if (TG == GPU)
  {
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, (float*)rand_colors_.data(),
                          rand_colors_.pitch() * rand_colors_.nrows() / sizeof(float));
  }
#endif

  traces_ = &traces1_;
  new_traces_ = &traces2_;
}


inline
__device__ i_float4 rand_color(int r)
{
  float v = 1.f;
  float s = 0.8f;

  int h = r;
  int hi = (h / 60) % 6;
  float f =  (h / 60.f) - floor(h / 60.f);
  float p = v * (1.0 - s);
  float q = v * (1.0 - (f*s));
  float t = v * (1.0 - ((1.0 - f) * s));

  switch (hi)
  {
    case 0: return i_float4(v, t, p, 1.f);
    case 1: return i_float4(q, v, p, 1.f);
    case 2: return i_float4(p, v, t, 1.f);
    case 3: return i_float4(p, q, v, 1.f);
    case 4: return i_float4(t, p, v, 1.f);
    default: return i_float4(v, p, q, 1.f);
  }
}

template <typename T>
inline __device__ void swap_kernel(T& a, T& b)
{
  T tmp = a;
  a = b;
  b = tmp;
}

template <typename V>
__device__ void draw_line2d(kernel_image2d<V> frame,
			    kernel_image2d<short> age_image,
			    point2d<int> a, point2d<int> b,
			    V color, short age)
{
  int x0 = a.col(); int y0 = a.row();
  int x1 = b.col(); int y1 = b.row();

  int steep = abs(y1 - y0) > abs(x1 - x0);

  if (steep)
  {
    swap_kernel(x0, y0);
    swap_kernel(x1, y1);
  }

  if (x0 > x1)
  {
    swap_kernel(x0, x1);
    swap_kernel(y0, y1);
  }

  int deltax = x1 - x0;
  int deltay = abs(y1 - y0);
  float error = 0.f;
  float deltaerr = deltay / float(deltax);
  int ystep;
  int y = y0;
  if (y0 < y1) ystep = 1; else ystep = -1;

  for (int x = x0; x <= x1; x++)
  {
    point2d<int> to_plot;
    if (steep)
      to_plot = point2d<int>(x, y);
    else
      to_plot = point2d<int>(y, x);

    if (age_image.has(to_plot))
    {
    age_image(to_plot) = age;
    frame(to_plot) = color;
    }

    error = error + deltaerr;
    if (error >= 0.5)
    {
      y = y + ystep;
      error = error - 1.0;
    }
  }
}


template <typename V>
__device__ void draw_line2d(kernel_image2d<V> frame,
			    point2d<int> a, point2d<int> b,
			    V color)
{
  int x0 = a.col(); int y0 = a.row();
  int x1 = b.col(); int y1 = b.row();

  int steep = abs(y1 - y0) > abs(x1 - x0);

  if (steep)
  {
    swap_kernel(x0, y0);
    swap_kernel(x1, y1);
  }

  if (x0 > x1)
  {
    swap_kernel(x0, x1);
    swap_kernel(y0, y1);
  }

  int deltax = x1 - x0;
  int deltay = abs(y1 - y0);
  float error = 0.f;
  float deltaerr = deltay / float(deltax);
  int ystep;
  int y = y0;
  if (y0 < y1) ystep = 1; else ystep = -1;

  for (int x = x0; x <= x1; x++)
  {
    point2d<int> to_plot;
    if (steep)
      to_plot = point2d<int>(x, y);
    else
      to_plot = point2d<int>(y, x);

    if (frame.has(to_plot))
    {
    frame(to_plot) = color;
    }

    error = error + deltaerr;
    if (error >= 0.5)
    {
      y = y + ystep;
      error = error - 1.0;
    }
  }
}

template <typename X>
__global__ void traj_decay(kernel_image2d<i_float4> frame,
			   kernel_image2d<short> age)
{
  point2d<int> p = thread_pos2d();
  if (!frame.has(p))
    return;

  if (age(p) >= 10)
  {
    frame(p) = i_float4(0.f, 0.f, 0.f, 1.f);
    age(p) = 0;
  }
  else if (age(p) >= 1)
    age(p)++;
}
/*
__global__ void update_trajectories(kernel_image2d<i_float4> rand_colors,
				    kernel_image2d<i_float4> colors,
				    kernel_image2d<i_float4> frame,
				    kernel_image2d<short> age,
				    kernel_image2d<tracker> track,
				    kernel_image2d<i_short2> matches,
				    unsigned age_filter)
{
  point2d<int> p = thread_pos2d();
  if (!track.has(p))
    return;

  point2d<int> src(p);
  point2d<int> dst(matches(p));
  if (!track.has(dst)) return;

  tracker tr = track(dst);
  rand_colors(p).w = 1.f;

  if (tr.to_draw)
  {
    draw_line2d(frame, age, src, dst, rand_color(tr.id), 1);
  }
  else
  {
    float rand_f = rand_colors(p).y;// (colors(p).x + colors(p).y / (0.001f+colors(p).z));
    int rand_ = int(2550*rand_f);// & 0x000000FF;

    if (tr.age == age_filter && tr.age
       && !((rand_+1) % 2300)
        )
    {
      int rand = 360 * rand_colors(p).x;
      int id = rand;

      frame(p) = rand_color(id);

      age(p) = 1;
      track(dst).to_draw = 1;
      track(dst).id = id;
    }
    else
      track(dst).to_draw = 0;
  }

}
*/
template <typename P, typename TR>
__global__ void update_trajectories(kernel_image2d<i_float4> rand_colors,
				    kernel_image2d<i_short2> matches,
				    kernel_image2d<TR> traces,
				    kernel_image2d<TR> new_traces,
				    kernel_image2d<P> particles,
				    kernel_image2d<i_float4> traj_,
				    kernel_image2d<short> age)
{
  point2d<int> p = thread_pos2d();
  if (!matches.has(p))
    return;

  point2d<int> src(p);

  P tr = particles(p);

  new_traces(p).reset();

  if (tr.age == 1) // New particle
  {
    rand_colors(p).w = 1.f;
    new_traces(p).color = rand_colors(p);
    new_traces(p).age = 1;
    new_traces(p).set_current_pos(i_int2(p));
    traj_(p) = rand_colors(p);
    age(p) = 1;
  }
  else
  {
    point2d<int> dst(matches(p));
    if (particles.has(dst) && matches(p) != i_int2(0,0))
    {
      if (traces(src).is_valid()) // Trace trajectory.
      {
	P tr2 = particles(dst);
	new_traces(dst) = traces(src);
	new_traces(dst).age += 1;
        new_traces(dst).set_current_pos(i_int2(dst));
	draw_line2d(traj_, age, src, dst, new_traces(dst).color, 1);
      }
    }
  }
}

template <typename V>
__device__ void draw_c8(kernel_image2d<V>& out, point2d<int>& p, const V& value)
{
  for_all_in_static_neighb2d(p, n, c4) if (out.has(n))
    out(n) = value;
}

template <typename X, typename TR>
__global__ void draw_traj_heads(kernel_image2d<TR> traces,
                                kernel_image2d<i_float4> out)
{
  point2d<int> p = thread_pos2d();
  if (!traces.has(p)) return;

  TR trace = traces(p);
  if (trace.is_valid())
  {
    out(p) = i_float4(1.f, 1.f, 1.f, 1.f);
    //draw_c8(out, p, i_float4(1.f, 0.f, 0.f, 1.f));
  }
}


template <typename P, typename TR>
__global__ void draw_straight_traj(kernel_image2d<P> particles,
                                   kernel_image2d<TR> traces,
                                   kernel_image2d<i_float4> out)
{
  point2d<int> p = thread_pos2d();
  if (!particles.has(p)) return;


  if (traces(p).age > 5)
  {
    for_all_in_static_neighb2d(p, n, c81)
      if (traces.has(n) && traces(p).age < traces(n).age)
        return;

    float dist = norml2(i_int2(particles(p).ipos) - i_int2(p));

    // if (dist > 10.f)
    if (out.has(traces(p).get_past_pos(8)))
    {
      for (unsigned i = 0; i < 8; i++)
        if (out.has(traces(p).get_past_pos(i)) && out.has(traces(p).get_past_pos(i+1)))
          draw_line2d(out, traces(p).get_past_pos(i), traces(p).get_past_pos(i+1), i_float4(0.f, 0.f, 0.f, 1.f));
          // out(traces(p).get_past_pos(i)) = i_float4(0.f, 0.f, 0.f, 1.f);
      // draw_line2d(out, p, i_int2(traces(p).get_past_pos(8)), i_float4(0.f, 0.f, 0.f, 1.f));

      // out(p) = i_float4(1.f, 1.f, 1.f, 1.f);
      draw_c8(out, p, i_float4(1.f, 0.f, 0.f, 1.f));
    }
  }
}

// template <typename P>
// __global__ void draw_straight_traj(kernel_image2d<P> particles,
//                                    kernel_image2d<trajectory_tracer::trace> traces,
//                                    kernel_image2d<i_float4> out)
// {
//   point2d<int> p = thread_pos2d();
//   if (!particles.has(p)) return;


//   if (particles(p).age > 5)
//   {
//     for_all_in_static_neighb2d(p, n, c81)
//       if (particles.has(n) && particles(p).age < particles(n).age)
//         return;

//     float dist = norml2(i_int2(particles(p).ipos) - i_int2(p));

//     // if (dist > 10.f)
//       draw_line2d(out, p, particles(p).ipos, i_float4(0.f, 0.f, 0.f, 1.f));

//     // out(p) = i_float4(1.f, 1.f, 1.f, 1.f);
//     draw_c8(out, p, i_float4(1.f, 0.f, 0.f, 1.f));
//   }
// }

template <target TG>
template <typename P>
inline
void
trajectory_tracer<TG>::update(const image2d_s2& matches,
                          const device_image2d<P>& particles)
{
  std::swap(traces_, new_traces_);
#ifndef NO_CUDA
  if (TG == GPU)
  {
    curandGenerateUniform(gen, (float*)rand_colors_.data(),
                          rand_colors_.pitch() * rand_colors_.nrows() / sizeof(float));
  }
#endif

  dim3 dimblock(16, 16);
  dim3 dimgrid = grid_dimension(rand_colors_.domain(), dimblock);

  update_trajectories<P, trace><<<dimgrid, dimblock>>>
    (rand_colors_, matches, *traces_, *new_traces_, particles, traj_, age_);

  traj_decay<int><<<dimgrid, dimblock>>>(traj_, age_);

  fill(straj_, i_float4(1.f, 1.f, 1.f, 1.f));
  draw_straight_traj<P><<<dimgrid, dimblock>>>(particles, *new_traces_, straj_);

}

template <target TG>
inline
void
trajectory_tracer<TG>::display(const std::string& w, const image2d_f1& colors)
{
  dim3 dimblock(16, 16);
  dim3 dimgrid = grid_dimension(rand_colors_.domain(), dimblock);

  image2d_f4 tmp(colors.domain());
  tmp = aggregate<float>::run(get_x(colors),get_x(colors),get_x(colors),1.f);

  fill(traj_heads_, i_float4(0.f, 0.f, 0.f, 1.f));
  draw_traj_heads<int><<<dimgrid, dimblock>>>(*new_traces_, traj_heads_);

  display_image_ = threshold(norml2(traj_heads_), 1.01f, traj_heads_, traj_);
  display_image_ = threshold(norml2(display_image_), 1.01f, display_image_,
                             tmp);

  // traj_heads_ = threshold(norml2(traj_heads_), 1.01f, traj_heads_, traj_);


  ImageView(w) <<= dg::dl() - traj_heads_ - display_image_ - straj_;
}

#endif
