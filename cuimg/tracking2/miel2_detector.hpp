#ifndef CUIMG_MIEL2_DETECTOR_HPP_
# define CUIMG_MIEL2_DETECTOR_HPP_

# include <cuimg/mt_apply.h>
# include <cuimg/memset.h>
# include <cuimg/neighb2d.h>

namespace cuimg
{

  namespace miel2
  {

    template <typename T>
    inline
    int get_dev(const T* data, int i, const int offsets[], int& min_diff)
    {
      int v1 = data[offsets[i]];
      int v2 = data[offsets[i+2]];
      //int dev = ::abs(*data - v1) + ::abs(*data - v2);
      int dev = ::abs(v2 - v1);
      min_diff = std::min(min_diff, dev);
      return dev;
    }

    template <typename I>
    inline int compute_saliency(i_short2 p, const I& in, float contrast_thresh, const int offsets[])
    {

      int min_diff = 999999;
      const typename I::value_type* data = &in(p);

      // Best order for queen_street.jpg : 4 0 2 6 3 1 5 7.
      get_dev(data, 4, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      get_dev(data, 0, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      get_dev(data, 2, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      get_dev(data, 6, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      get_dev(data, 3, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      get_dev(data, 1, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      get_dev(data, 5, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      get_dev(data, 7, offsets, min_diff); if (min_diff < contrast_thresh) return 0;

      if (min_diff >= contrast_thresh)
      {
	return (min_diff);
      }
      else
	return 0;
    }
  }

  miel2_1s_detector::miel2_1s_detector(const obox2d& d)
    : saliency_(d),
      contrast_(d),
      new_points_(d),
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

#ifndef NO_CPP0X
  template <typename J>
  void
  miel2_1s_detector::update(const host_image2d<gl8u>& input, const J& mask)
  {
    START_PROF(miel2_compute_saliency);

    int offsets[16];
    for (unsigned i = 0; i < 16; i ++)
    {
      i_int2 o = circle_r3_h[i];
      offsets[i] = (int(input.pitch()) * o.r()) / sizeof(gl8u) + o.c();
    }

    cv::Mat opencv_s2(input_s2_);
    cv::GaussianBlur(cv::Mat(input), opencv_s2, cv::Size(11, 11), 1, 1, cv::BORDER_REPLICATE);
    fill_border_clamp(input_s2_);

    dim3 dimblock = ::cuimg::dimblock(cpu(), sizeof(i_uchar1), input.domain());
    // local_jet_static_<0, 0, 1, 1>::run(input, input_s2_, tmp_, 0, dimblock);
    mt_apply2d(sizeof(i_float1), input.domain() - border(8),
	       [this, &input, &offsets] (i_int2 p)
							 {
							   saliency_(p) = miel2::compute_saliency(p, input_s2_, contrast_th_, offsets);
							 }, cpu());

    END_PROF(miel2_compute_saliency);
  }

  template <typename F, typename PS>
  void
  miel2_1s_detector::new_particles(const F& feature, PS& pset_)
  {
    SCOPE_PROF(miel2_new_particles_detector);

    int offsets[8];
    for (unsigned i = 0; i < 8; i ++)
    {
      i_int2 o = c8_h[i];
      offsets[i] = (int(saliency_.pitch()) * o.r()) / sizeof(saliency_(0,0)) + o.c();
    }

    memset(new_points_, 0);
    typename PS::kernel_type pset = pset_;
    mt_apply2d(sizeof(i_float1), saliency_.domain() / 5,
               [this, &feature, &pset, &offsets] (i_int2 p)
               {
		 p = (p) * 5 + i_int2(3,3);
		 float vmin = saliency_(p);
		 i_int2 min_p = p;
		 for_all_in_static_neighb2d(p, n, c8_h)
		   if (vmin < saliency_(n)) { vmin = saliency_(n); min_p = n; }

                 //if (pset_.has(min_p)) return;
                 if (saliency_(min_p) < this->contrast_th_) return;

                 // if (this->saliency_(p) < this->contrast_th_) return;
		 // auto* data = &this->saliency_(p);
                 // for (int i = 0; i < 8; i++)
                 // {
                 //   i_int2 n(p + i_int2(c8_h[i]));
                 //   if (*data < data[offsets[i]])
                 //     return;
                 // }

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
