#ifndef CUIMG_GRADIENT_DESCENT_MATCHER_H_
# define CUIMG_GRADIENT_DESCENT_MATCHER_H_

# include <cuimg/improved_builtin.h>

namespace cuimg
{

  template <typename F, typename FI>
  inline __host__ __device__
  i_short2 gradient_descent_match(i_short2 prediction, F f, FI feature_img);

  template <typename S, typename FI>
  inline __host__ __device__
  i_int2 naive_local_match(i_short2 prediction,
			   const S& sample,
			   const FI& feature_img);

}

# include <cuimg/tracking2/gradient_descent_matcher.hpp>

#endif
