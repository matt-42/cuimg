#ifndef CUIMG_TRANSLATION_STABILIZER_HPP_
# define CUIMG_TRANSLATION_STABILIZER_HPP_

# include <cuimg/memset.h>

namespace cuimg
{

  inline
  translation_stabilizer::translation_stabilizer()
    : correction_(0, 0),
      speed_(0, 0),
      histo(300, 300)
  {
  }

  inline
  translation_stabilizer::~translation_stabilizer()
  {
  }

  template <typename V>
  inline void
  translation_stabilizer::update(const V& particles)
  {
    float K = 0.05f;
    float acc = 0.1f;

    if (correction_.x < 0)
      speed_.x += acc;
    else if (correction_.x > 0)
      speed_.x -= acc;

    if (correction_.y < 0)
      speed_.y += acc;
    else if (correction_.y > 0)
      speed_.y -= acc;

    if (std::abs(-correction_.x * K) < std::abs(speed_.x))
      speed_.x = -correction_.x * K;
    if (std::abs(-correction_.y * K) < std::abs(speed_.y))
      speed_.y = -correction_.y * K;

    correction_ += speed_;

    hmax_ = 0;
    i_int2 tr(0,0);
    memset(histo, 0);
    int hhs = histo.nrows() / 2;

    for (unsigned i = 0; i < particles.size(); i++)
    {
      const auto& p = particles[i];
      if (p.age > 5 && p.pos != i_int2(-1, -1))
      {
        i_int2 speed = p.pos - p.pos_history[0];
        i_int2 bin(hhs + speed.r(), hhs + speed.c());
        if (histo.has(bin) && (histo(bin) += p.age) > hmax_)
        {
          tr = speed;
          hmax_ = histo(bin);
        }
      }
    }

    correction_ -= tr;

  }

  inline i_float2
  translation_stabilizer::correction() const
  {
    return correction_;
  }

  inline const host_image2d<int>&
  translation_stabilizer::histogram() const
  {
    return histo;
  }


  inline const int&
  translation_stabilizer::histogram_max() const
  {
    return hmax_;
  }

}

# include <cuimg/tracking/translation_stabilizer.hpp>

#endif // !CUIMG_TRANSLATION_STABILIZER_H_
