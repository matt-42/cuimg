#ifndef CUIMG_DENSE_DETECTOR_HPP_
# define CUIMG_DENSE_DETECTOR_HPP_

# include <cuimg/mt_apply.h>

namespace cuimg
{

  dense_detector::dense_detector(const obox2d& d)
  {
  }

	template <typename J>
  void
  dense_detector::update(const host_image2d<gl8u>& input, const J& mask)
  {
  }

#ifndef NO_CPP0X
  template <typename F, typename PS>
  void
  dense_detector::new_particles(const F& feature, PS& pset_)
  {
    SCOPE_PROF(mdfl_new_particles_detector);
    typename PS::kernel_type pset = pset_;
    st_apply2d(sizeof(i_float1), feature.domain() - border(0),
               [this, &feature, &pset, &pset_] (i_int2 p)
               {
                 if (!pset.has(p))
									 pset_.add(p, feature(p));
               }, cpu());
  }
#endif

}

#endif
