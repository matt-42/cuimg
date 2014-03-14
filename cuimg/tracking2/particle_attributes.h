#ifndef CUIMG_TRACKING_PARTICLE_ATTRIBUTES_H
# define CUIMG_TRACKING_PARTICLE_ATTRIBUTES_H


#include <vector>



namespace cuimg
{

  template <typename T>
  class particle_attributes : public std::vector<T>
  {
    particle_attributes();

  private:
    std::vector<T> tmp_;
  };

  template <typename T, typename TR>
  void synchronize_with_tracker(particle_attributes<T>& v, const TR& tr);

}



#endif
