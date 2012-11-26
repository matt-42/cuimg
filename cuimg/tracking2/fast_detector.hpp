#ifndef CUIMG_FAST_DETECTOR_HPP_
# define CUIMG_FAST_DETECTOR_HPP_

# include <cuimg/mt_apply.h>

namespace cuimg
{

  namespace fast
  {

    template <typename I>
    inline gl8u compute_saliency(i_short2 p, const I& in, int scale, float contrast_thresh)
    {
      unsigned char pv = in(p).x;
      unsigned char max_contrast = 0.f;
      for(int i = 0; i < 16; i++)
      {
        unsigned char v1 = in(p + i_int2(circle_r3_h[i]) * scale).x;
        unsigned char contrast = ::abs(pv - v1);
        if (max_contrast < contrast) max_contrast = contrast;
      }

      if (max_contrast >= contrast_thresh)
        return max_contrast;
      else
        return 0;
    }
  }

  fast_detector::fast_detector(const obox2d& d)
    : n_(9),
      saliency_(d),
      new_points_(d)
  {
  }

  fast_detector&
  fast_detector::set_contrast_threshold(float f)
  {
    contrast_th_ = f;
    return *this;
  }

  fast_detector&
  fast_detector::set_fast_threshold(float f)
  {
    fast_th_ = f;
    return *this;
  }

  fast_detector&
  fast_detector::set_n(unsigned n)
  {
    n_ = n;
    return *this;
  }

  void
  fast_detector::update(const host_image2d<gl8u>& input)
  {
    input_ = input;
    START_PROF(fast_compute_saliency);
    mt_apply2d(sizeof(i_float1), input.domain() - border(8),
               [this, &input] (i_int2 p)
               {
                 saliency_(p) = fast::compute_saliency(p, input, 1, contrast_th_);
               }, arch::cpu());

    END_PROF(fast_compute_saliency);

    // if (input.nrows() == 480)
    //   DISPLAY(saliency_);
  }

  template <typename F, typename PS>
  void
  fast_detector::new_particles(const F& feature, PS& pset)
  {
    SCOPE_PROF(fast_new_particles_detector);
    memset(new_points_, 0);
    mt_apply2d(sizeof(i_float1), saliency_.domain() - border(8),
               [this, &feature, &pset] (i_int2 p)
               {
                 if (pset.has(p)) return;
                 if (saliency_(p) < contrast_th_) return;

                 for (int i = 0; i < 8; i++)
                 {
                   i_int2 n(p + i_int2(c8[i]));
                   if (saliency_(p) < saliency_(n) || pset.has(n))
                     return;
                 }

                 unsigned max_n = 0;
                 unsigned n = 0;
                 bool status = 2;
                 gl8u vp = input_(p);
                 for (int i = 0; i < 16; i++)
                 {
                   gl8u vn = input_(p + i_int2(circle_r3_h[i]));
                   int sign = int(vn) > int(vp);
                   unsigned char dist = ::abs(int(vn) - int(vp));
                   if (dist > fast_th_)
                   {
                     if (sign == status) n++;
                     else
                     {
                       if (n > max_n)
                         max_n = n;
                       status = sign;
                       n = 1;
                     }
                   }
                   else
                     status = 2;
                 }

                 if (n != 16 && status != 2)
                 {
                   int i = 0;
                   while (true)
                   {
                     gl8u vn = input_(p + i_int2(circle_r3_h[i]));
                     int sign = int(vn) > int(vp);
                     unsigned char dist = ::abs(int(vn) - int(vp));

                     if (dist <= fast_th_ || sign != status) break;

                     i++;
                     assert(i < 16);
                   }


                 }

                 if (n < n_) return;

                 new_points_(p) = 1;
               }, arch::cpu());

    st_apply2d(sizeof(i_float1), saliency_.domain() - border(8),
               [this, &feature, &pset] (i_int2 p)
               {
                 if (new_points_(p)) pset.add(p, feature(p));
               }, arch::cpu());

  }

}

#endif
