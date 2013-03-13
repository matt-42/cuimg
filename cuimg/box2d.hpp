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

  int
  box2d::size() const
  {
    return nrows() * ncols();
  }

  const i_int2
  box2d::center() const
  {
    return (a_ + b_) / 2;
  }

  bool box2d::has(const point2d<int>& p) const
  {
    return p.row() >= a_.r() && p.row() <= b_.r() &&
      p.col() >= a_.c() && p.col() <= b_.c();
  }


  void box2d::extend(const point2d<int>& p)
  {
    if (p.row() < a_.r()) a_.r() = p.row();
    if (p.row() > b_.r()) b_.r() = p.row();

    if (p.col() < a_.c()) a_.c() = p.col();
    if (p.col() > b_.c()) b_.c() = p.col();
  }


  void box2d::extend(const box2d& bb)
  {
    extend(bb.a_);
    extend(bb.b_);
  }

}

#endif




