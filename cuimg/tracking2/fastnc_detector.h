#ifndef CUIMG_FASTNC_DETECTOR_H_
# define CUIMG_FASTNC_DETECTOR_H_

# include <cuimg/improved_builtin.h>
# include <cuimg/cpu/host_image2d.h>

namespace cuimg
{

  template <typename A>
  class fastnc_detector
  {
  public:
    typedef typename A::template image2d<gl8u>::ret image2d_gl8u;
    typedef typename A::template image2d<i_short2>::ret image2d_short2;

    inline fastnc_detector(const obox2d& d);

    template <typename J>
    inline void update(const image2d_gl8u& input, const J& mask);

    template <typename F, typename PS>
    inline void new_particles(F& feature, PS& pset);
    template <typename F, typename PS>
    inline void new_particles(F& feature, PS& pset, const cpu&);

#ifndef NO_CUDA
    template <typename F, typename PS>
    inline void new_particles(F& feature, PS& pset, const cuda_gpu&);
#endif

    inline fastnc_detector& set_fast_threshold(float f);
    inline fastnc_detector& set_n(unsigned n);

    inline const image2d_gl8u& saliency() { return saliency_; }
    inline const image2d_gl8u& contrast() { return contrast_; }

  private:
    float contrast_th_;
    float fast_th_;
    unsigned n_;
    image2d_gl8u saliency_;
    image2d_gl8u contrast_;
    image2d_gl8u input_;
    image2d_short2 new_points_;
  };

}

# include <cuimg/tracking2/fastnc_detector.hpp>

#endif
