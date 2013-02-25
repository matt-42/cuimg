#ifndef CUIMG_OBOX2D_HPP_
# define CUIMG_OBOX2D_HPP_

# include <cuimg/obox2d.h>

namespace cuimg
{
  obox2d::obox2d()
  {
  }

  obox2d::obox2d(unsigned nrows, unsigned ncols)
    : nrows_(nrows),
      ncols_(ncols)
  {
  }

  obox2d::obox2d(const obox2d& img)
    : nrows_(img.nrows()),
      ncols_(img.ncols())
  {
  }

  obox2d&
  obox2d::operator=(const obox2d& d)
  {
    nrows_ = d.nrows();
    ncols_ = d.ncols();
    return *this;
  }

  unsigned obox2d::nrows() const
  {
    return nrows_;
  }

  unsigned obox2d::ncols() const
  {
    return ncols_;
  }


  unsigned obox2d::size() const
  {
    return ncols_ * nrows_;
  }

  bool obox2d::has(const point2d<int>& p) const
  {
    return p.row() >= 0 && p.row() < point2d<int>::coord(nrows_) &&
      p.col() >= 0 && p.col() < point2d<int>::coord(ncols_);
  }

  point2d<int> obox2d::mod(const point2d<int>& p) const
  {
    point2d<int> res = p;
    while (res.r() < 0) res.r() += nrows();
    while (res.c() < 0) res.c() += ncols();
    res.r() = res.r() % nrows();
    res.c() = res.c() % ncols();
    assert(has(res));
    return res;
  }


  obox2d_iterator::obox2d_iterator(i_int2 p, obox2d b) 
    : p_(p), 
      line_width_(b.ncols()) 
  {}

  obox2d_iterator::obox2d_iterator(const obox2d_iterator& mit) 
    : p_(mit.p_) 
  {}

  obox2d_iterator&
  obox2d_iterator::operator++()
  {
    if (p_.c() != (line_width_ - 1))
      p_.c()++;
    else
      p_ = i_int2(p_.r() + 1, 0);
    return *this;
  }
  obox2d_iterator
  obox2d_iterator::operator++(int)
  {
    obox2d_iterator tmp(*this);
    operator++();
    return tmp;
  }

  bool
  obox2d_iterator::operator==(const obox2d_iterator& rhs)
  {
    return p_ == rhs.p_;
  }

  bool
  obox2d_iterator::operator!=(const obox2d_iterator& rhs)
  {
    return p_ != rhs.p_;
  }

  const i_int2&
  obox2d_iterator::operator*()
  {
    return p_;
  }

}

#endif
