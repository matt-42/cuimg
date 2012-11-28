#ifndef CUIMG_FILTER_SPACIAL_INCOHERENCES_H_
# define CUIMG_FILTER_SPACIAL_INCOHERENCES_H_

# include <cuimg/improved_builtin.h>

namespace cuimg
{

  template <typename PS>
  inline __host__ __device__
  bool is_spacial_incoherence(const PS& pset, i_short2 p)
  {
    /* assert(pset.has(p)); */
    /* assert(pset(p).age > 0); */

    int bad = 0;
    int good = 0;

    for(int i = 0; i < 49; i++)
    {
      point2d<int> n(p.r() + c49[i][0],
                     p.c() + c49[i][1]);

      if (pset.has(n) && pset(n).age >= 1)
      {
        if (norml2(pset(n).speed -
                   pset(p).speed) > 3.f)
          bad++;
        else good++;
      }
    }

    if ((good + bad) > 0)
      return (float(bad) / (good + bad)) > 0.6f;
    else
      return false;
  }

}

# include <cuimg/tracking2/gradient_descent_matcher.hpp>

#endif
