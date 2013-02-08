#ifndef CUIMG_MERGE_TRAJECTORIES_H_
# define CUIMG_MERGE_TRAJECTORIES_H_

# include <cuimg/improved_builtin.h>
# include <cuimg/builtin_math.h>

namespace cuimg
{

  template <typename PI>
  inline __host__ __device__
  void merge_trajectories(PI& pset, particle& part)
  {
    i_int2 p = part.pos;
    for (int i = 0; i < 8; i++)
    {
      i_int2 n(p + i_int2(c8[i]));
      if (pset.domain().has(n) and pset.has(n))
      {
    	particle& buddy = pset(n);
    	if (buddy.age > (part.age + 2) and norml2(part.speed - buddy.speed) < 2.f)
    	{
    	  part.age = 0;
    	  break;
    	}
      }
    }

    // particle& buddy = pset(part.pos);
    // if (buddy.age > part.age and norml2(part.speed - buddy.speed) < 2.f)
    //   part.age = 0;
  }

}

# include <cuimg/tracking2/gradient_descent_matcher.hpp>

#endif
