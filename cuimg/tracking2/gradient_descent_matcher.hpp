#ifndef CUIMG_GRADIENT_DESCENT_MATCHER_H_
# define CUIMG_GRADIENT_DESCENT_MATCHER_H_

namespace cuimg
{

  template <typename F, typename FI>
  inline __host__ __device__
  i_short2 gradient_descent_match(i_short2 prediction, F f, FI feature_img)
  {
    unsigned match_i = 8;
    for (int search = 0; search < 7; search++)
    {
      int i = c8_it[match_i][0];
      int end = c8_it[match_i][1];
      {
	i_int2 n(prediction + c8[i]);
	{
	  float d = feature_img.distance(f, n);
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
	i_int2 n(prediction + c8[i]);
	{
	  float d = feature_img.distance(f, n);
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

    return match;

  }

  template <typename F, typename S>
  __device__ i_int2 naive_local_match(i_short2 prediction,
				       const S& sample,
				       const F& feature_img)
  {
    for_all_in_static_neighb2d(prediction, n, c49)
      if (n->row() > 10 && n->row() < (particles.domain().nrows() - 10) &&
	  n->col() > 10 && n->col() < (particles.domain().ncols() - 10))
      {
	float d = feature_img.distance_linear(sample, n);
	if (d < match_distance)
	{
	  match = n;
	  match_distance = d;
	}
      }
  }

}

#endif
