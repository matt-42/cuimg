#ifndef CUIMG_MIEL3_DETECTOR_HPP_
# define CUIMG_MIEL3_DETECTOR_HPP_

# include <cuimg/mt_apply.h>
# include <cuimg/memset.h>

namespace cuimg
{

  namespace miel3
  {

    template <typename T>
    inline
    int get_dev(const T* data, int i, int j, const int offsets[], int& min_diff)
    {
      int v1 = data[offsets[i]];
      int v2 = data[offsets[j]];
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
      // get_dev(data, 4, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 0, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 2, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 6, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 3, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 1, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 5, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 7, offsets, min_diff); if (min_diff < contrast_thresh) return 0;

      // get_dev(data, 4, 13, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 6, 11, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 3, 14, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 6, 10, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 1, 7, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 9, 15, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 5, 12, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 4, 14, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 9, 0, offsets, min_diff); if (min_diff < contrast_thresh) return 0;


      // get_dev(data, 0,15 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 10,14 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 7,10 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 1,7 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 4,13 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 6,11 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 6,10 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 4,8 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 5,10 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 0,5 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 1,9 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 3,14 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 3,12 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 5,13 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 8,15 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 4,12 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;


      // get_dev(data, 1,15 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 3,14 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 1,7 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 5,12 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 6,10 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 9,15 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 4,13 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 6,11 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 6,8 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 5,11 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 0,13 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 13,10 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;


      // get_dev(data, 0, 15, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 10, 14, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 9, 8, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 5,12 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 8,0 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 6,11 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 6,2 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 3,12 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;

      // get_dev(data, 1, 15, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 9, 15, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 6, 11, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 4,7 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 4,12 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 5,10 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 2,15 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 3,7 , offsets, min_diff); if (min_diff < contrast_thresh) return 0;



      // // Sigma 3
      // get_dev(data, 0, 15, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 2, 6, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 6, 11, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 7, 10, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 5, 8, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 4, 9, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 4, 13, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 3, 12, offsets, min_diff); if (min_diff < contrast_thresh) return 0;

      // get_dev(data, 1, 7, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 9, 15, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 3, 13, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 5, 11, offsets, min_diff); if (min_diff < contrast_thresh) return 0;

      // get_dev(data, 0, 8, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 4, 12, offsets, min_diff); if (min_diff < contrast_thresh) return 0;


      // get_dev(data, 1, 7, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 9, 15, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 3, 13, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 5, 11, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 0, 8, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      // get_dev(data, 4, 12, offsets, min_diff); if (min_diff < contrast_thresh) return 0;


      get_dev(data, 1, 11, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      get_dev(data, 9, 3, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      get_dev(data, 3, 13, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      get_dev(data, 5, 11, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      get_dev(data, 5, 15, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      get_dev(data, 7, 13, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      get_dev(data, 7, 1, offsets, min_diff); if (min_diff < contrast_thresh) return 0;

      if (min_diff >= contrast_thresh)
      {
	return (min_diff);
      }
      else
	return 0;
    }
  }

  miel3_1s_detector::miel3_1s_detector(const obox2d& d)
    : saliency_(d),
      contrast_(d),
      new_points_(d),
      input_s2_(d, 3),
      tmp_(d)
  {
  }

  miel3_1s_detector::miel3_1s_detector(const miel3_1s_detector& d)
  {
    *this = d;
  }

  miel3_1s_detector&
  miel3_1s_detector::set_contrast_threshold(float f)
  {
    contrast_th_ = f;
    return *this;
  }

#ifndef NO_CPP0X
  template <typename J>
  void
  miel3_1s_detector::update(const host_image2d<gl8u>& input, const J& mask)
  {
    START_PROF(miel3_compute_saliency);

    int offsets[16];
    for (unsigned i = 0; i < 16; i ++)
    {
      i_int2 o = circle_r3_h[i];
      offsets[i] = (int(input.pitch()) * o.r()) / sizeof(gl8u) + o.c();
    }

    cv::Mat opencv_s2(input_s2_);
    cv::GaussianBlur(cv::Mat(input), opencv_s2, cv::Size(11, 11), 1, 1, cv::BORDER_REPLICATE);
    //copy(input, input_s2_);
    fill_border_clamp(input_s2_);
    dim3 dimblock = ::cuimg::dimblock(cpu(), sizeof(i_uchar1), input.domain());
    // local_jet_static_<0, 0, 1, 1>::run(input, input_s2_, tmp_, 0, dimblock);
    mt_apply2d(sizeof(i_float1), input.domain() - border(0),
	       [this, &input, &offsets] (i_int2 p)
							 {
							   saliency_(p) = miel3::compute_saliency(p, input_s2_, contrast_th_, offsets);
							 }, cpu());

    END_PROF(miel3_compute_saliency);
  }

  template <typename F, typename PS>
  void
  miel3_1s_detector::new_particles(const F& feature, PS& pset_)
  {
    SCOPE_PROF(miel3_new_particles_detector);

    int offsets[8];
    for (unsigned i = 0; i < 8; i ++)
    {
      i_int2 o = c8_h[i];
      offsets[i] = (int(saliency_.pitch()) * o.r()) / sizeof(saliency_(0,0)) + o.c();
    }

    memset(new_points_, 0);
    typename PS::kernel_type pset = pset_;
    mt_apply2d(sizeof(i_float1), saliency_.domain() - border(8),
               [this, &feature, &pset, &offsets] (i_int2 p)
               {
                 if (this->saliency_(p) < this->contrast_th_) return;
		 auto* data = &this->saliency_(p);
                 for (int i = 0; i < 8; i++)
                 {
                   i_int2 n(p + i_int2(c8_h[i]));
                   if (*data < data[offsets[i]])
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
