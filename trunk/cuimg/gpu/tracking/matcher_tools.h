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
  __global__ void filter_robbers(i_short2* particles_vec,
                                 unsigned n_particles,
                                 kernel_image2d<T> particles,
                                 kernel_image2d<T> new_particles,
                                 kernel_image2d<i_short2> matches)
  {
    unsigned threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid >= n_particles) return;
    point2d<int> p = particles_vec[threadid];
    // point2d<int> p = thread_pos2d();
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


  template <typename T>
  __global__ void extract_age(kernel_image2d<T> particles,
                                 kernel_image2d<i_float4> disp)
  {
    point2d<int> p = thread_pos2d();
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

  template <typename T>
  __global__ void filter_false_matching(i_short2* particles_vec,
                                        unsigned n_particles,
                                        kernel_image2d<T> particles,
                                        kernel_image2d<T> new_particles,
                                        kernel_image2d<i_short2> matches,
                                        kernel_image2d<char> errors
                                        ,kernel_image2d<i_float4> disp
                                        )

  {
    unsigned threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid >= n_particles) return;
    point2d<int> p = particles_vec[threadid];
    // point2d<int> p = thread_pos2d();

    if (!particles.has(p))
      return;

    errors(p) = 0;
    disp(p) = i_float4(0.f, 0.f, 0.f, 1.f);

    if (new_particles(p).age == 0) return;

    int bad = 0;
    int good = 0;

    //for_all_in_static_neighb2d(p, n, c25)
    for(int i = 0; i < 25; i++)
    {
      point2d<int> n(p.row() + c25[i][0],
                     p.col() + c25[i][1]);
    /* for(int i = 0; i < 49; i++) */
    /* { */
    /*   point2d<int> n(p.row() + c49[i][0], */
    /*                  p.col() + c49[i][1]); */
      if (new_particles.has(n) && new_particles(n).age >= 1)
      {
        if (norml2(new_particles(n).speed -
                   new_particles(p).speed) > 3.f)
          bad++;
        else good++;
      }
    }
    if (float(bad) / (good + bad) > 0.6f)
    {
      new_particles(p).age = 0;
      errors(p) = 1;
      if (new_particles(p).age <= 1)
        disp(p) = i_float4(1.f, 0.f, 0.f, 1.f);
    }
  }

  template <typename T>
  __global__ void init_particles_kernel(kernel_image2d<T> particles)
  {
    point2d<int> p = thread_pos2d();
    if (!particles.has(p))
      return;
    particles(p).age = 0;
  }


  template <typename F, typename T>
  __global__ void create_particles_kernel_(F f,
                                         kernel_image2d<T> particles,
                                         kernel_image2d<i_float1> pertinence,
                                         kernel_image2d<i_float4> test_)
  {
    point2d<int> p = thread_pos2d();
    if (!particles.has(p) || particles(p).age != 0)
      return;

    if (pertinence(p).x > 0.6f)
    {
      particles(p).age = 1;
      particles(p).fault = 0;
      particles(p).state = f.current_frame()(p);
      particles(p).speed = i_float2(0.f, 0.f);
      particles(p).acceleration = i_float2(0.f, 0.f);
      test_(p) = i_float4(0.f, 1.f, 0.f, 1.f);
    }
    else
      test_(p) = i_float4(1.f, 0.f, 0.f, 1.f);

  }


  template <typename F, typename T>
  __global__ void create_particles_kernel(F f,
                                         kernel_image2d<T> particles,
                                         kernel_image2d<i_float1> pertinence,
                                         kernel_image2d<i_float4> test_)
  {
    point2d<int> p = thread_pos2d();
    if (!particles.has(p) || p.row() < 12 || p.row() > particles.domain().nrows() - 12 ||
        p.col() < 12 || p.col() > particles.domain().ncols() - 12)
    return;

    float pp = pertinence(p).x;
    if (pp < 0.6f)
      return;

    if (particles(p).age != 0)
      return;

    {
      bool max = true;
      //for_all_in_static_neighb2d(p, n, c8)
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
        np.state = f.new_state(p);
        np.speed = i_float2(0.f, 0.f);

        particles(p) = np;
      }
      else
      {}
    }

  }

}

#endif // ! CUIMG_MATCHER_TOOLS_H_
