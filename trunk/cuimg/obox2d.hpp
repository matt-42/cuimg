#ifndef CUIMG_OBOX2D_HPP_
# define CUIMG_OBOX2D_HPP_

# include <cuimg/obox2d.h>

namespace cuimg
{
  template <typename P>
  obox2d<P>::obox2d()
  {
  }

  template <typename P>
  obox2d<P>::obox2d(unsigned nrows, unsigned ncols)
    : nrows_(nrows),
      ncols_(ncols)
  {
  }

  template <typename P>
  obox2d<P>::obox2d(const obox2d<P>& img)
    : nrows_(img.nrows()),
      ncols_(img.ncols())
  {
  }

  template <typename P>
  obox2d<P>&
  obox2d<P>::operator=(const obox2d<P>& d)
  {
    nrows_ = d.nrows();
    ncols_ = d.ncols();
    return *this;
  }

  template <typename P>
  unsigned obox2d<P>::nrows() const
  {
    return nrows_;
  }

  template <typename P>
  unsigned obox2d<P>::ncols() const
  {
    return ncols_;
  }

  template <typename P>
  bool obox2d<P>::has(const point& p) const
  {
    return p.row() >= 0 && p.row() < point::coord(nrows_) &&
      p.col() >= 0 && p.col() < point::coord(ncols_);
  }

}

#endif
