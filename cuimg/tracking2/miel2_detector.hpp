#ifndef CUIMG_MIEL2_DETECTOR_HPP_
# define CUIMG_MIEL2_DETECTOR_HPP_

# include <cuimg/mt_apply.h>
# include <cuimg/memset.h>

namespace cuimg
{

  namespace miel2
  {

    template <typename I>
    inline int compute_saliency(i_short2 p, const I& in, float contrast_thresh)
    {
      int min_diff = 999999;
      int pv = in(p).x;
      for(int i = 0; i < 8; i++)
      {
        int v1 = in(p + i_int2(circle_r3_h[i])).x;
				int v2 = in(p + i_int2(circle_r3_h[i+8])).x;
        // int v3 = in(p + i_int2(circle_r3_h[i]) * 0.5).x;
				// int v4 = in(p + i_int2(circle_r3_h[i+8]) * 0.5).x;
				// int dev = ::abs(pv - v1) + ::abs(pv - v2) + ::abs(pv - v3) + ::abs(pv - v4);
				// min_diff = std::min(min_diff, dev);
				int dev = (pv - v1) * (pv - v1) +
					(pv - v2) * (pv - v2);
				min_diff = std::min(min_diff, dev);
      }

			min_diff /= 4;
			if (min_diff >= contrast_thresh)
				return (min_diff);
			else
				return 0;
    }
	}

  miel2_1s_detector::miel2_1s_detector(const obox2d& d)
    : saliency_(d),
      contrast_(d),
      new_points_(d),
      saliency_mode_(MAX),
      input_s2_(d),
      tmp_(d)
  {
  }


  miel2_1s_detector::miel2_1s_detector(const miel2_1s_detector& d)
  {
    *this = d;
  }

  miel2_1s_detector&
  miel2_1s_detector::set_contrast_threshold(float f)
  {
    contrast_th_ = f;
    return *this;
  }


  miel2_1s_detector&
  miel2_1s_detector::set_dev_threshold(float f)
  {
    dev_th_ = f;
    return *this;
  }

  miel2_1s_detector&
  miel2_1s_detector::set_saliency_mode(saliency_mode m)
  {
    saliency_mode_ = m;
    return *this;
  }

#ifndef NO_CPP0X
  template <typename J>
  void
  miel2_1s_detector::update(const host_image2d<gl8u>& input, const J& mask)
  {
    START_PROF(miel2_compute_saliency);

    dim3 dimblock = ::cuimg::dimblock(cpu(), sizeof(i_uchar1), input.domain());
    // local_jet_static_<0, 0, 1, 1>::run(input, input_s2_, tmp_, 0, dimblock);
    mt_apply2d(sizeof(i_float1), input.domain() - border(8),
							 [this, &input] (i_int2 p)
							 {
								 saliency_(p) = miel2::compute_saliency(p, input, contrast_th_);
							 }, cpu());

    END_PROF(miel2_compute_saliency);
  }

  template <typename F, typename PS>
  void
  miel2_1s_detector::new_particles(const F& feature, PS& pset_)
  {
    SCOPE_PROF(miel2_new_particles_detector);
    memset(new_points_, 0);
    typename PS::kernel_type pset = pset_;
    mt_apply2d(sizeof(i_float1), saliency_.domain() - border(8),
               [this, &feature, &pset] (i_int2 p)
               {
                 if (pset.has(p)) return;
                 if (this->contrast_(p) < this->contrast_th_) return;
                 if (this->saliency_(p) < this->dev_th_) return;
                 for (int i = 0; i < 8; i++)
                 {
                   i_int2 n(p + i_int2(c8_h[i]));
                   if (this->saliency_(p) < this->saliency_(n) || pset.has(n))
                     return;
                 }

                 this->new_points_(p) = 1;
               }, cpu());

    st_apply2d(sizeof(char), saliency_.domain() - border(8),
               [this, &feature, &pset_] (i_int2 p)
               {
                 if (this->new_points_(p)) pset_.add(p, feature(p));
               }, cpu());

  }

#endif

}

#endif
