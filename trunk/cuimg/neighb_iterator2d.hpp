#ifndef CUIMG_NEIGHB_ITERATOR2D_HPP_
# define CUIMG_NEIGHB_ITERATOR2D_HPP_

# include <cuimg/neighb_iterator2d.h>

namespace cuimg
{
  template <typename N>
  neighb_iterator2d<N>::neighb_iterator2d(const point& p, const N& n)
    : cur_(p),
      n_(n),
      i_(0)
  {
    invalidate();
  }

  template <typename N>
  bool neighb_iterator2d<N>::is_valid() const
  {
    return i_ <= n_.size();
  }

  template <typename N>
  void neighb_iterator2d<N>::invalidate()
  {
     i_ = n_.size() + 1;
  }

  template <typename N>
  void neighb_iterator2d<N>::next()
  {
      p_ = i_int2(cur_) + i_int2(n_[i_]);
      i_++;
  }

  template <typename N>
  void neighb_iterator2d<N>::start()
  {
    i_ = 0;
    p_ = i_int2(cur_) + i_int2(n_[i_]);
    i_++;
  }

  template <typename N>
  const typename neighb_iterator2d<N>::point& neighb_iterator2d<N>::operator*() const
  {
      return p_;
  }

  template <typename N>
  const typename neighb_iterator2d<N>::point* neighb_iterator2d<N>::operator->() const
  {
      return &p_;
  }

 // template <typename N>
 // neighb_iterator2d<N>::operator point2d<int>() const
 // {
 //     return p_;
 // }

}

#endif
