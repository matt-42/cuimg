#ifndef CUIMG_OBOX3D_HPP_
# define CUIMG_OBOX3D_HPP_

# include <cuimg/obox3d.h>

namespace cuimg
{
  template <typename P>
  obox3d<P>::obox3d(unsigned nslices, unsigned nrows, unsigned ncols)
    : nslices_(nslices),
      nrows_(nrows),
      ncols_(ncols)
  {
  }

  template <typename P>
  obox3d<P>::obox3d(const obox3d<P>& img)
    : nslices_(img.nslices()),
      nrows_(img.nrows()),
      ncols_(img.ncols())
  {
  }

  template <typename P>
  obox3d<P>&
  obox3d<P>::operator=(const obox3d<P>& d)
  {
    nrows_ = d.nrows();
    ncols_ = d.ncols();
    return *this;
  }

  template <typename P>
  unsigned obox3d<P>::nslices() const
  {
    return nslices_;
  }

  template <typename P>
  unsigned obox3d<P>::nrows() const
  {
    return nrows_;
  }

  template <typename P>
  unsigned obox3d<P>::ncols() const
  {
    return ncols_;
  }

  template <typename P>
  bool obox3d<P>::has(const point& p) const
  {
    return p.sli() >= 0 && p.sli() < point::coord(nslices_) &&
           p.row() >= 0 && p.row() < point::coord(nrows_) &&
           p.col() >= 0 && p.col() < point::coord(ncols_);
  }

}

#endif
