#ifndef CUIMG_STATIC_NEIGHB2D_HPP_
# define CUIMG_STATIC_NEIGHB2D_HPP_

# include <cuimg/static_neighb2d.h>

namespace cuimg
{

 // template <unsigned S>
 // static_neighb2d<S>::static_neighb2d(const int n[S][2])


  template <unsigned S>
  unsigned static_neighb2d<S>::size() const
  {
    return S;
  }

  template <unsigned S>
  const typename static_neighb2d<S>::point& static_neighb2d<S>::operator[](unsigned s) const
  {
    return *(point2d<int>*)(dpoints_[s]);
  }

  template <unsigned S>
  static_neighb2d<S>::static_neighb2d(const static_neighb2d<S>& n)
    : dpoints_(n.dpoints_)
  {
  }

  template <unsigned S>
  static_neighb2d<S>& static_neighb2d<S>::operator=(const static_neighb2d<S>& n)
  {
    dpoints_ = n.dpoints_;
    return *this;
  }

}

#endif
