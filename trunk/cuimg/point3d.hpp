#ifndef CUIMG_POINT3D_HPP_
# define CUIMG_POINT3D_HPP_

namespace cuimg
{

  template <typename C>
  point3d<C>::point3d()
  {
  }

  template <typename C>
  point3d<C>::point3d(C sli, C row, C col)
  {
    coords_[0] = sli;
    coords_[1] = row;
    coords_[2] = col;
  }

  template <typename C>
  point3d<C>::point3d(const point3d<C>& d)
  {
    coords_[0] = d.sli();
    coords_[1] = d.row();
    coords_[2] = d.col();
  }

  template <typename C>
  point3d<C>&
  point3d<C>::operator=(const point3d<C>& d)
  {
    coords_[0] = d.sli();
    coords_[1] = d.row();
    coords_[2] = d.col();
    return *this;
  }

  template <typename C>
  C
  point3d<C>::sli() const
  {
    return coords_[0];
  }

  template <typename C>
  C
  point3d<C>::row() const
  {
    return coords_[1];
  }

  template <typename C>
  C
  point3d<C>::col() const
  {
    return coords_[2];
  }

  template <typename C>
  template <typename T>
  point3d<C>::point3d(const improved_builtin<T, 3>& bt)
  {
    coords_[0] = bt.x;
    coords_[1] = bt.y;
    coords_[2] = bt.z;
  }

  template <typename C>
  point3d<C>::operator typename make_bt<C, 3>::ret() const
  {
    return typename make_bt<C, 3>::ret(sli(), row(), col());
  }

}

#endif
