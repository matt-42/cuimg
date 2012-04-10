#ifndef CUIMG_POINT2D_HPP_
# define CUIMG_POINT2D_HPP_

namespace cuimg
{

  template <typename C>
  inline
  point2d<C>::point2d()
  {
  }

  template <typename C>
  template <typename D>
  inline
  point2d<C>::point2d(const improved_builtin<D, 2>& bt)
    : super(bt)
  {
  }

  template <typename C>
  inline
  point2d<C>::point2d(C row, C col)
    : super(row, col)
  {
  }

  template <typename C>
  template <typename D>
  inline
  point2d<C>&
  point2d<C>::operator=(const improved_builtin<D, 2>& d)
  {
    super::operator=(d);
    return *this;
  }

  template <typename C> inline C  point2d<C>::row() const { return this->x; }
  template <typename C> inline C  point2d<C>::col() const { return this->y; }
  template <typename C> inline C& point2d<C>::row()       { return this->x; }
  template <typename C> inline C& point2d<C>::col()       { return this->y; }

}

#endif
