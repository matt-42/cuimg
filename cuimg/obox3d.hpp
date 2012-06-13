#ifndef CUIMG_OBOX3D_HPP_
# define CUIMG_OBOX3D_HPP_

# include <cuimg/obox3d.h>

namespace cuimg
{

  obox3d::obox3d()
  {
  }

  obox3d::obox3d(unsigned nslices, unsigned nrows, unsigned ncols)
    : nslices_(nslices),
      nrows_(nrows),
      ncols_(ncols)
  {
  }

  obox3d::obox3d(const obox3d& img)
    : nslices_(img.nslices()),
      nrows_(img.nrows()),
      ncols_(img.ncols())
  {
  }

  obox3d&
  obox3d::operator=(const obox3d& d)
  {
    nrows_ = d.nrows();
    ncols_ = d.ncols();
    return *this;
  }

  unsigned obox3d::nslices() const
  {
    return nslices_;
  }

  unsigned obox3d::nrows() const
  {
    return nrows_;
  }

  unsigned obox3d::ncols() const
  {
    return ncols_;
  }

  bool obox3d::has(const point3d<int>& p) const
  {
    return p.sli() >= 0 && p.sli() < point3d<int>::coord(nslices_) &&
           p.row() >= 0 && p.row() < point3d<int>::coord(nrows_) &&
           p.col() >= 0 && p.col() < point3d<int>::coord(ncols_);
  }

}

#endif
