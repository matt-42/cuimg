#ifndef CUIMG_FAST_DETECTOR_H_
# define CUIMG_FAST_DETECTOR_H_

# include <cuimg/improved_builtin.h>
# include <cuimg/cpu/host_image2d.h>

namespace cuimg
{

  template <typename A>
  class fast_detector
  {
  public:
    typedef typename A::template image2d<gl8u>::ret image2d_gl8u;
    typedef typename A::template image2d<i_short2>::ret image2d_short2;

    inline fast_detector(const obox2d& d);

    inline void update(const host_image2d<gl8u>& input);
    inline void update(const device_image2d<gl8u>& input);

    template <typename F, typename PS>
    inline void new_particles(const F& feature, PS& pset);
    template <typename F, typename PS>
    inline void new_particles(const F& feature, PS& pset, const cpu&);

#ifndef NO_CUDA
    template <typename F, typename PS>
    inline void new_particles(const F& feature, PS& pset, const cuda_gpu&);
#endif

    inline fast_detector& set_fast_threshold(float f);
    inline fast_detector& set_n(unsigned n);

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

# include <cuimg/tracking2/fast_detector.hpp>

#endif
