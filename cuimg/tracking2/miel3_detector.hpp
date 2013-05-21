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
      int v = data[0];
      int v1 = data[offsets[i]];
      int v2 = data[offsets[j]];
      int dev = ::abs(v2 + v1 - 2 * v) / 2;
      min_diff = std::min(min_diff, dev);
      return dev;
    }

    template <typename I>
    inline int compute_saliency(i_short2 p, const I& in, float contrast_thresh, const int offsets[])
    {

      int min_diff = 999999;
      const typename I::value_type* data = &in(p);

      get_dev(data, 1, 11, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      get_dev(data, 9, 3, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      get_dev(data, 3, 13, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      get_dev(data, 5, 11, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      get_dev(data, 5, 15, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      get_dev(data, 7, 13, offsets, min_diff); if (min_diff < contrast_thresh) return 0;
      get_dev(data, 7, 1, offsets, min_diff); if (min_diff < contrast_thresh) return 0;

      return min_diff;
    }
  }

  miel3_1s_detector::miel3_1s_detector(const obox2d& d)
    : saliency_(d),
      contrast_(d),
      new_points_(d),
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

    dim3 dimblock = ::cuimg::dimblock(cpu(), sizeof(i_uchar1), input.domain());
    mt_apply2d(sizeof(i_float1), input.domain() - border(0),
	       [this, &input, &offsets] (i_int2 p)
	       {
		 saliency_(p) = miel3::compute_saliency(p, input, contrast_th_, offsets);
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

    mt_apply2d(sizeof(i_float1), saliency_.domain() / 3,
               [this, &feature, &pset] (i_int2 p)
               {
		 p = (p) * 3 + i_int2(1,1);
		 float vmin = saliency_(p);
		 i_int2 min_p = p;
		 for_all_in_static_neighb2d(p, n, c8_h)
		   if (vmin < saliency_(n)) { vmin = saliency_(n); min_p = n; }

                 if (pset.has(min_p)) return;
                 if (saliency_(min_p) == 0) return;

                 new_points_(min_p) = 1;
               }, cpu());

    // mt_apply2d(sizeof(i_float1), saliency_.domain() - border(0),
    //            [this, &feature, &pset, &offsets] (i_int2 p)
    //            {
    //              if (this->saliency_(p) < this->contrast_th_) return;
    // 		 auto* data = &this->saliency_(p);
    //              for (int i = 0; i < 8; i++)
    //              {
    //                i_int2 n(p + i_int2(c8_h[i]));
    //                if (*data < data[offsets[i]])
    //                  return;
    //              }

    //              this->new_points_(p) = 1;
    //            }, cpu());

    st_apply2d(sizeof(char), saliency_.domain() - border(0),
               [this, &feature, &pset_] (i_int2 p)
               {
                 if (this->new_points_(p)) pset_.add(p, feature(p));
               }, cpu());

  }

#endif

}

#endif
