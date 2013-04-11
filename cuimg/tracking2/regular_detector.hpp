#ifndef CUIMG_REGULAR_DETECTOR_HPP_
# define CUIMG_REGULAR_DETECTOR_HPP_

# include <cuimg/mt_apply.h>
# include <cuimg/memset.h>

namespace cuimg
{

  regular_detector::regular_detector(const obox2d& d)
    : saliency_(d),
      contrast_(d),
      new_points_(d),
      step_(1),
      grad_th_(1)
  {
  }

  regular_detector::regular_detector(const regular_detector& d)
  {
    *this = d;
  }

  regular_detector&
  regular_detector::set_step(int s)
  {
    step_ = s;
    return *this;
  }


  regular_detector&
  regular_detector::set_grad_th(int s)
  {
    grad_th_ = s;
    return *this;
  }

#ifndef NO_CPP0X
  template <typename J>
  void
  regular_detector::update(const host_image2d<gl8u>& input, const J& mask)
  {
    START_PROF(regular_compute_saliency);

    mt_apply2d(sizeof(i_float1), input.domain() - border(1),
               [this, &input, &mask] (i_int2 p)
	       {
		 const int d = 2;
		 int res = 0;
		 if ((input.domain()).has(p))
		 {
		   int vp = input(p);
		   for (int i = 0; i < 8; i++)
		   {
		     gl8u vn = input(p + i_int2(c8_h[i])*2);
		     res += ::abs(vn - vp);
		   }
		   // res = ::abs(int(input_(p + i_int2(0,d))) - int(input_(p + i_int2(0,-d)))) +
		   //   ::abs(int(input_(p + i_int2(-d,0))) - int(input_(p + i_int2(d,0)))) +
		   //   ::abs(int(input_(p + i_int2(-d,-d))) - int(input_(p + i_int2(d,d)))) +
		   //   ::abs(int(input_(p + i_int2(-d,d))) - int(input_(p + i_int2(d,-d))))
		   //   ;
		 }
		 contrast_(p) = res / 8;
	       }, cpu()
	       );


    END_PROF(regular_compute_saliency);
  }

  template <typename F, typename PS>
  void
  regular_detector::new_particles(const F& feature, PS& pset_)
  {
    SCOPE_PROF(regular_new_particles_detector);

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
                 if ((p.r() % this->step_) or (p.c() % this->step_)) return;
		 if (contrast_(p) < grad_th_) return;
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
