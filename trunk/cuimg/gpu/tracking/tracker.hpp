#ifndef CUIMG_TRACKER_HPP_
# define CUIMG_TRACKER_HPP_


namespace cuimg
{

  template <typename SA, typename F>
  tracker<SA, F>::tracker(SA& sa, F& f)
    : f_(f),
      sa_(sa)
  {
  }

  template <typename SA, typename F>
  template <typename V>
  void tracker<SA, F>::update(const image2d<V>& in)
  {
    f_.update(in);
    sa_.update(f_);
  }

}

#endif // !CUIMG_TRACKER_HPP_
