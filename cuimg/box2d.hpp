#ifndef CUIMG_BOX2D_HPP_
# define CUIMG_BOX2D_HPP_

# include <cuimg/box2d.h>

namespace cuimg
{
  box2d::box2d()
  {
  }

  box2d::box2d(i_int2 a, i_int2 b)
    : a_(a),
      b_(b)
  {
  }

  box2d::box2d(const box2d& box)
    : a_(box.a_),
      b_(box.b_)
  {
  }

  box2d&
  box2d::operator=(const box2d& d)
  {
    a_ = d.a_;
    b_ = d.b_;
    return *this;
  }

  unsigned box2d::nrows() const
  {
    return b_.r() - a_.r();
  }

  unsigned box2d::ncols() const
  {
    return b_.c() - a_.c();
  }

  const i_int2&
  box2d::p1() const
  {
    return a_;
  }

  const i_int2&
  box2d::p2() const
  {
    return b_;
  }

  bool box2d::has(const point2d<int>& p) const
  {
    return p.row() >= a_.r() && p.row() <= b_.r() &&
      p.col() >= a_.c() && p.col() <= b_.c();
  }

}

#endif
