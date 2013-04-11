#ifndef CUIMG_GRADIENT_DETECTOR_HPP_
# define CUIMG_GRADIENT_DETECTOR_HPP_

# include <cuimg/mt_apply.h>
# include <cuimg/memset.h>

namespace cuimg
{

  gradient_detector::gradient_detector(const obox2d& d)
    : saliency_(d),
      input_s2_(d),
      contrast_(d),
      new_points_(d),
      grad_th_(1),
      sigma_(0.f)
  {
  }

  gradient_detector::gradient_detector(const gradient_detector& d)
  {
    *this = d;
  }


  gradient_detector&
  gradient_detector::set_sigma(float s)
  {
    sigma_ = s;
    return *this;
  }

  gradient_detector&
  gradient_detector::set_grad_th(int s)
  {
    grad_th_ = s;
    return *this;
  }

#ifndef NO_CPP0X
  template <typename J>
  void
  gradient_detector::update(const host_image2d<gl8u>& input, const J& mask)
  {
    START_PROF(gradient_compute_saliency);

    if (sigma_ > 0.f)
      cv::GaussianBlur(cv::Mat(input), cv::Mat(input_s2_), cv::Size(7, 7), sigma_, sigma_, cv::BORDER_REPLICATE);
    else
      copy(input, input_s2_);

    mt_apply2d(sizeof(i_float1), input.domain() - border(1),
               [this, &input, &mask] (i_int2 p)
	       {
		 const int d = 2;
		 int res = 0;
		 int smin = 0;
		 int smax = 0;
		 if ((input.domain()).has(p))
		 {
		   int vp = input_s2_(p);
		   for (int i = 0; i < 8; i++)
		   {
		     gl8u vn = input_s2_(p + i_int2(c8_h[i])*2);
		     if (vn < vp)
		       smin += ::abs(vn - vp);
		     else
		       smax += ::abs(vn - vp);
		     res += ::abs(vn - vp);
		   }
		   // res = ::abs(int(input_(p + i_int2(0,d))) - int(input_(p + i_int2(0,-d)))) +
		   //   ::abs(int(input_(p + i_int2(-d,0))) - int(input_(p + i_int2(d,0)))) +
		   //   ::abs(int(input_(p + i_int2(-d,-d))) - int(input_(p + i_int2(d,d)))) +
		   //   ::abs(int(input_(p + i_int2(-d,d))) - int(input_(p + i_int2(d,-d))))
		   //   ;
		 }
		 contrast_(p) = res / 8;
		 //contrast_(p) = std::max(smin, smax) / 8;
	       }, cpu()
	       );


    END_PROF(gradient_compute_saliency);
  }

  template <typename F, typename PS>
  void
  gradient_detector::new_particles(const F& feature, PS& pset_)
  {
    SCOPE_PROF(gradient_new_particles_detector);

    int offsets[8];
    for (unsigned i = 0; i < 8; i ++)
    {
      i_int2 o = c8_h[i];
      offsets[i] = (int(saliency_.pitch()) * o.r()) / sizeof(saliency_(0,0)) + o.c();
    }

    memset(new_points_, 0);
    typename PS::kernel_type pset = pset_;
    mt_apply2d(sizeof(i_float1), feature.domain(),
               [this, &feature, &pset, &offsets] (i_int2 p)
               {
		 if (contrast_(p) < grad_th_) return;
                 for (int i = 0; i < 8; i++)
                 {
                   i_int2 n(p + i_int2(c8_h[i]));
                   if (contrast_(p) < contrast_(n) || pset.has(n))
                     return;
                 }

                 this->new_points_(p) = 1;
               }, cpu());

    st_apply2d(sizeof(char), feature.domain() - border(0),
               [this, &feature, &pset_] (i_int2 p)
               {
                 if (this->new_points_(p)) pset_.add(p, feature(p));
               }, cpu());

  }

#endif

}

#endif
