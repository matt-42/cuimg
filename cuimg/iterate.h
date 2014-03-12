#ifndef CUIMG_ITERATE_HH
# define CUIMG_ITERATE_HH

# include <cuimg/box2d.h>
# include <cuimg/obox2d.h>

namespace cuimg
{
  template <typename C>
  struct iterate_
  {
  public:
    inline iterate_(const C& c) : container(c) {}
    const C& container;
  };


  template <typename D>
  inline
  iterate_<D> iterate(const D& d)
  {
    return iterate_<D>(d);
  }


  template <typename F>
  void operator>>(F f, iterate_<box2d> it)
  {
    for (unsigned r = it.container.p1().r(); r < it.container.p2().r(); r++)
      for (unsigned c = it.container.p1().c(); c < it.container.p2().c(); c++)
        f(i_int2(r, c));
  }

  template <typename F>
  void operator>>(F f, iterate_<obox2d> it)
  {
    for (unsigned r = 0; r < it.container.nrows(); r++)
      for (unsigned c = 0; c < it.container.ncols(); c++)
        f(i_int2(r, c));
  }

  template <typename F, typename T>
  void operator>>(F f, iterate_<std::vector<T> > it)
  {
    for (unsigned i = 0; i < it.container.size(); i++)
      f(it.container[i]);
  }

}

#endif
