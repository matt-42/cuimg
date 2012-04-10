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

  bool obox2d::has(const point& p) const
  {
    return p.row() >= 0 && p.row() < point::coord(nrows_) &&
      p.col() >= 0 && p.col() < point::coord(ncols_);
  }

}

#endif
