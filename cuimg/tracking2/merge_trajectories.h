#ifndef CUIMG_MERGE_TRAJECTORIES_H_
# define CUIMG_MERGE_TRAJECTORIES_H_

# include <cuimg/improved_builtin.h>

namespace cuimg
{

  template <typename PI>
  inline __host__ __device__
  bool merge_trajectories(PI& pset, int i)
  {
    /* assert(i < pset.dense_particles().size()); */
    /* particle& part = pset.dense_particles()[i]; */

    /* particle& buddy = pset.sparse_particles()(part.pos); */
    /* if (buddy.age < part.age) */
    /* { */
    /*   buddy.set(part, i); */
    /*   return true; */
    /* } */
    /* else */
    /* { */
    /*   if (buddy.vpos != i) */
    /* 	part.age = 0; */
    /* } */
    return false;
  }

}

# include <cuimg/tracking2/gradient_descent_matcher.hpp>

#endif
