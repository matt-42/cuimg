#ifndef CUIMG_TRACKING_PARTICLE_ATTRIBUTES_HPP
# define CUIMG_TRACKING_PARTICLE_ATTRIBUTES_HPP


#include <cuimg/tracking2/particle_attributes.hpp>

namespace cuimg
{

  template <typename T>
  particle_attributes::particle_attributes()
  {
  }

  template <typename T, typename TR>
  void synchronize_with_tracker(particle_attributes<T>& v, const TR& tr)
  {
    for(unsigned i = 0; i < tr.pset.dense_particles.size(); i++)
  }

}



#endif
