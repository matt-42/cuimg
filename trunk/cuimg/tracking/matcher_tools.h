#ifndef CUIMG_MATCHER_TOOLS_H_
# define CUIMG_MATCHER_TOOLS_H_

# include <cuimg/gpu/kernel_image2d.h>

namespace cuimg
{

#define check_robbers_sig(TG, T)                \
  kernel_image2d<T> ,                           \
    kernel_image2d<T> ,                         \
    kernel_image2d<i_short2> ,                  \
    kernel_image2d<i_float4>,                   \
    &check_robbers<TG, T>

  template <unsigned target, typename T>
  __host__ __device__ void check_robbers(thread_info<target> ti,
                                         kernel_image2d<T> particles,
                                         kernel_image2d<T> new_particles,
                                         kernel_image2d<i_short2> matches,
                                         kernel_image2d<i_float4> test_)
  {
    point2d<int> p = thread_pos2d(ti);
    if (!particles.has(p))
      return;

    if (particles(p).age == 0) test_(p) = i_float4(0.f, 0.f, 0.f, 1.f); // No particle here
    else if (!new_particles.has(matches(p))) test_(p) = i_float4(0.f, 0.f, 1.f, 1.f); // No match found
    else if (new_particles(matches(p)).age < particles(p).age) test_(p) = i_float4(1.f, 0.f, 0.f, 1.f);
    else test_(p) = i_float4(0.f, 1.f, 0.f, 1.f);
  }

#define filter_robbers_sig(TG, T)               \
  i_short2*,                                    \
    unsigned ,                                  \
    kernel_image2d<T> ,                         \
    kernel_image2d<T> ,                         \
    kernel_image2d<i_short2>,                   \
    &filter_robbers<TG, T>

  template <unsigned target, typename T>
  __host__ __device__ void filter_robbers(thread_info<target> ti,
                                          i_short2* particles_vec,
                                          unsigned n_particles,
                                          kernel_image2d<T> particles,
                                          kernel_image2d<T> new_particles,
                                          kernel_image2d<i_short2> matches)
  {
    unsigned threadid = ti.blockIdx.x * ti.blockDim.x + ti.threadIdx.x;
    if (threadid >= n_particles) return;
    point2d<int> p = particles_vec[threadid];
    // point2d<int> p = thread_pos2d(ti);
    if (!particles.has(p))
      return;

    if (particles(p).age == 0) return;

    if (new_particles.has(matches(p)))
    {
      if (new_particles(matches(p)).age < particles(p).age)
      {
        new_particles(matches(p)) = particles(p);
      }
    }

  }

#define extract_age_sig(TG, T)                  \
  kernel_image2d<T> ,                           \
    kernel_image2d<i_float4>,                   \
    &extract_age<TG, T>


  template <unsigned target, typename T>
  __host__ __device__ void extract_age(thread_info<target> ti,
                              kernel_image2d<T> particles,
                              kernel_image2d<i_float4> disp)
  {
    point2d<int> p = thread_pos2d(ti);
    if (!particles.has(p))
      return;

    if (particles(p).age == 0)
    {
      disp(p) = i_float4(0.f, 0.f, 0.f, 1.f);
      return;
    }

    float v = particles(p).age / 50.f;

    if (v >= 1.f) v = 1.f;


    disp(p) = i_float4(1.f, 0.f, 0.f, 1.f) * v +
      i_float4(0.f, 1.f, 0.f, 1.f) * (1.f - v);

    return;
  }

#define filter_false_matching_sig(TG, T)        \
  i_short2* ,                                   \
    unsigned ,                                  \
    kernel_image2d<T>,                          \
    kernel_image2d<T>,                          \
    kernel_image2d<i_short2>,                   \
    kernel_image2d<char>,                       \
    kernel_image2d<i_float4>,                   \
    &filter_false_matching<TG, T>

  template <unsigned target, typename T>
  __host__ __device__ void filter_false_matching(thread_info<target> ti,
                                        i_short2* particles_vec,
                                        unsigned n_particles,
                                        kernel_image2d<T> particles,
                                        kernel_image2d<T> new_particles,
                                        kernel_image2d<i_short2> matches,
                                        kernel_image2d<char> errors
                                        ,kernel_image2d<i_float4> disp
                                        )
  {
    unsigned threadid = ti.blockIdx.x * ti.blockDim.x + ti.threadIdx.x;
    if (threadid >= n_particles) return;
    point2d<int> p = particles_vec[threadid];
    // point2d<int> p = thread_pos2d(ti);

    if (!particles.has(p))
      return;

    point2d<int> match = matches(p);
    if (!particles.has(match)) return;
    disp(match) = i_float4(0.f, 0.f, 0.f, 1.f);

    if (new_particles(match).age == 0) return;

    int bad = 0;
    int good = 0;

    //for_all_in_static_neighb2d(p, n, c25)
    for(int i = 0; i < 25; i++)
    {
      point2d<int> n(match.row() + c25[i][0],
                     match.col() + c25[i][1]);
    /* for(int i = 0; i < 49; i++) */
    /* { */
    /*   point2d<int> n(p.row() + c49[i][0], */
    /*                  p.col() + c49[i][1]); */
      if (new_particles.has(n) && new_particles(n).age >= 1)
      {
        if (norml2(new_particles(n).speed -
                   new_particles(match).speed) > 3.f)
          bad++;
        else good++;
      }
    }
    if (float(bad) / (good + bad) > 0.6f)
    {
      new_particles(match).age = 0;
      errors(match) = 1;
      if (new_particles(match).age <= 1)
        disp(match) = i_float4(1.f, 0.f, 0.f, 1.f);
    }
  }

#define init_particles_kernel_sig(TG, T)        \
  kernel_image2d<T>,                            \
    &init_particles_kernel<TG, T>

  template <unsigned target, typename T>
  __host__ __device__ void init_particles_kernel(thread_info<target> ti,
                                                 kernel_image2d<T> particles)
  {
    point2d<int> p = thread_pos2d(ti);
    if (!particles.has(p))
      return;
    particles(p).age = 0;
  }


#define create_particles_kernel__sig(TG, F, T)  \
  F,                                            \
    kernel_image2d<T>,                          \
    kernel_image2d<i_float1>,                   \
    kernel_image2d<i_float4>,                   \
    kernel_image2d<typename F::feature_t> ,     \
    &create_particles_kernel_<TG, F, T>         \

  template <unsigned target, typename F, typename T>
  __host__ __device__ void create_particles_kernel_(thread_info<target> ti,
                                           F f,
                                           kernel_image2d<T> particles,
                                           kernel_image2d<i_float1> pertinence,
                                           kernel_image2d<typename F::feature_t> states,
                                           kernel_image2d<i_float4> test_)
  {
    point2d<int> p = thread_pos2d(ti);
    if (!particles.has(p) || particles(p).age != 0)
      return;

    if (pertinence(p).x > 0.6f)
    {
      particles(p).age = 1;
      particles(p).fault = 0;
      particles(p).speed = i_float2(0.f, 0.f);
      particles(p).acceleration = i_float2(0.f, 0.f);
      states(p) = f.new_state(p);
      test_(p) = i_float4(0.f, 1.f, 0.f, 1.f);
    }
    else
      test_(p) = i_float4(1.f, 0.f, 0.f, 1.f);

  }


#define create_particles_kernel_sig(TG, F, T)       \
  F,                                                \
    kernel_image2d<T> ,                             \
    kernel_image2d<i_float1> ,                      \
    kernel_image2d<F::feature_t> ,                  \
    &create_particles_kernel<TG, F, T>

  template <unsigned target, typename F, typename T>
  __host__ __device__ void create_particles_kernel(thread_info<target> ti,
                                                   F f,
                                                   kernel_image2d<T> particles,
                                                   kernel_image2d<i_float1> pertinence,
                                                   kernel_image2d<typename F::feature_t> states)
  {
    point2d<int> p = thread_pos2d(ti);
    if (!particles.has(p) || p.row() < 12 || p.row() > particles.domain().nrows() - 12 ||
        p.col() < 12 || p.col() > particles.domain().ncols() - 12)
    return;

    float pp = pertinence(p).x;

    /* if (particles(p).age != 0) */
    /* { */
    /*   if (pp < 0.1f) */
    /* 	particles(p).age = 0; */
    /*   return; */
    /* } */

    if (pp < 0.7f)
      return;

    {
      bool max = true;
      //for_all_in_static_neighb2d(p, n, c8)
#pragma unroll
      for (int i = 0; i < 8; i++)
      {
        point2d<int> n(p.row() + c8[i][0],
                       p.col() + c8[i][1]);

        if (pp < pertinence(n).x || particles(n).age > 1)
        {
          max = false;
          break;
        }
      }

      if (max)
      {
        T np;
        np.ipos = i_int2(p);
        np.age = 1;
        // np.state = f.new_state(p);
        np.speed = i_float2(0.f, 0.f);

        states(p) = f.new_state(p);
        particles(p) = np;
      }
      else
      {}
    }

  }

}

#endif // ! CUIMG_MATCHER_TOOLS_H_
