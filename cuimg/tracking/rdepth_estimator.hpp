#ifndef CUIMG_RDEPTH_ESTIMATOR_HPP_
# define CUIMG_RDEPTH_ESTIMATOR_HPP_

# include <cuimg/memset.h>

namespace cuimg
{

  inline
  rdepth_estimator::rdepth_estimator()
  {
  }

  inline
  rdepth_estimator::~rdepth_estimator()
  {
  }

  template <typename V>
  inline void
  rdepth_estimator::update(V& particles)
  {

    for(unsigned i = 0; i < particles.size(); i++)
    {
      auto& p = particles[i];
      i_float2 pos = p.pos - foe_;
      int age = p.age >= 3 ? (3) : p.age;
      i_float2 speed =i_float2(p.pos_history[age-1] - p.pos) / float(age);
      double d;
      bool to_far = false;
      if (norml2(speed) < .4f)
      {
	d = 10000.f;
      }
      else
	d = norml2(pos) / norml2(speed);

      if (p.age < 1)
	p.depth = d;
      else
	p.depth = p.depth * 0.9f + d * 0.1f;
    }

  }

  inline void
  rdepth_estimator::set_foe(const i_short2& f)
  {
    foe_ = f;
  }

}

#endif // !CUIMG_RDEPTH_ESTIMATOR_H_
