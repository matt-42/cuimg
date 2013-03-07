#ifndef CUIMG_DENSE_DETECTOR_HPP_
# define CUIMG_DENSE_DETECTOR_HPP_

# include <cuimg/mt_apply.h>

namespace cuimg
{

  dense_detector::dense_detector(const obox2d& d)
  {
  }

  void
  dense_detector::update(const host_image2d<gl8u>& input)
  {
  }

#ifndef NO_CPP0X
  template <typename F, typename PS>
  void
  dense_detector::new_particles(const F& feature, PS& pset)
  {
    SCOPE_PROF(mdfl_new_particles_detector);
    st_apply2d(sizeof(i_float1), feature.domain() - border(8),
               [this, &feature, &pset] (i_int2 p)
               {
                 if (!pset.has(p))
		   pset.add(p, feature(p));
               }, cpu());
  }
#endif

}

#endif
