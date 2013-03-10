#ifndef CUIMG_GRADIENT_DESCENT_MATCHER_HPP_
# define CUIMG_GRADIENT_DESCENT_MATCHER_HPP_

#include <cuimg/neighb2d.h>

namespace cuimg
{

  template <typename F, typename FI>
  inline __host__ __device__ 
  i_short2 gradient_descent_match(i_short2 prediction, F f, FI& feature_img, float& distance, unsigned scale = 1)
  {
    typedef typename FI::architecture A;
    i_short2 match = prediction;
    float match_distance = feature_img.distance(f, prediction);
    unsigned match_i = 8;
    box2d domain = feature_img.domain() - border(8);

    if (!domain.has(prediction))
    {
      distance = 999999.f;
      return prediction;
    }
    assert(domain.has(prediction));
    for (int search = 0; search < 7; search++)
    {
      int i = arch_neighb2d<A>::get(c8_it_h, c8_it, match_i)[0];
      int end = arch_neighb2d<A>::get(c8_it_h, c8_it, match_i)[1];
      {
        i_int2 n(prediction + i_int2(arch_neighb2d<A>::get(c8_h, c8, i)));
        {
          float d = feature_img.distance(f, n, scale);
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
      for(; i != end; i = (i + 1) & 7)
      {
        i_int2 n(prediction + i_int2(arch_neighb2d<A>::get(c8_h, c8, i)));
        {
          float d = feature_img.distance(f, n, scale);
          if (d < match_distance)
          {
            match = n;
            match_i = i;
            match_distance = d;
          }
        }
      }

      if (i_int2(prediction) == i_int2(match) || !domain.has(match))
        break;
      else
        prediction = match;

    }

    distance = match_distance;
    return match;

  }


  template <typename F, typename FI>
  inline __host__ __device__ 
  i_short2 gradient_descent_match2(i_short2 prediction, F f, FI& feature_img, float& distance, unsigned scale = 1)
  {
    i_short2 match = prediction;
    float match_distance = feature_img.distance(f, prediction);
    unsigned match_i = 8;
    box2d domain = feature_img.domain() - border(7);

    if (!domain.has(prediction)) return prediction;
    assert(domain.has(prediction));
    for (int search = 0; search < 7; search++)
    {
      for(int i = 0; i != 25; i++)
      {
        i_int2 n(prediction + i_int2(c25[i]));
        {
          float d = feature_img.distance(f, n, scale);
          if (d < match_distance)
          {
            match = n;
            match_i = i;
            match_distance = d;
          }
        }
      }

      if (i_int2(prediction) == i_int2(match) || !domain.has(match))
        break;
      else
        prediction = match;

    }

    distance = match_distance;
    return match;

  }

  template <typename F, typename FI>
  inline __host__ __device__ 
  i_short2 two_step_gradient_descent_match(i_short2 prediction, F f, FI& feature_img, float& distance)
  {
    i_short2 match1 = gradient_descent_match(prediction, f, feature_img, distance, 2);
    return gradient_descent_match(match1, f, feature_img, distance, 1);
  }

  template <typename F, typename S>
  inline __host__ __device__ 
  i_int2 naive_local_match(i_short2 prediction,
                           const S& sample,
                           const F& feature_img)
  {
    i_short2 match = prediction;
    float match_distance = feature_img.distance(sample, prediction);

    for_all_in_static_neighb2d(prediction, n, c49)
      if (n->row() > 10 && n->row() < (feature_img.domain().nrows() - 10) &&
          n->col() > 10 && n->col() < (feature_img.domain().ncols() - 10))
      {
        float d = feature_img.distance_linear(sample, n);
        if (d < match_distance)
        {
          match = i_int2(n);
          match_distance = d;
        }
      }

    return match;
  }

}

#endif
