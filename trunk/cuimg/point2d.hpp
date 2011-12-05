#ifndef CUIMG_POINT2D_HPP_
# define CUIMG_POINT2D_HPP_

namespace cuimg
{

  template <typename C>
  point2d<C>::point2d()
  {
  }

  template <typename C>
  point2d<C>::point2d(C row, C col)
  {
    coords_[0] = row;
    coords_[1] = col;
  }

  template <typename C>
  point2d<C>::point2d(const point2d<C>& d)
  {
    coords_[0] = d.row();
    coords_[1] = d.col();
  }

  template <typename C>
  point2d<C>&
  point2d<C>::operator=(const point2d<C>& d)
  {
    coords_[0] = d.row();
    coords_[1] = d.col();
    return *this;
  }

  template <typename C>
  C
  point2d<C>::row() const
  {
    return coords_[0];
  }

  template <typename C>
  C
  point2d<C>::col() const
  {
    return coords_[1];
  }


  template <typename C>
  C&
  point2d<C>::row()
  {
    return coords_[0];
  }

  template <typename C>
  C&
  point2d<C>::col()
  {
    return coords_[1];
  }

  template <typename C>
  template <typename T>
  point2d<C>::point2d(const improved_builtin<T, 2>& bt)
  {
    coords_[0] = bt.x;
    coords_[1] = bt.y;
  }

  template <typename C>
  point2d<C>::operator typename make_bt<C, 2>::ret() const
  {
    return typename make_bt<C, 2>::ret(row(), col());
  }

  template <typename C, typename D>
  bool operator==(const point2d<C>& a, const point2d<D>& b)
  {
    return a.row() == b.row() && a.col() == b.col();
  }

}

#endif
