#ifndef CUIMG_ITERATE_HH

namespace cuimg
{
  template <typename C>
  struct iterate_
  {
  public:
    inline iterate_(C& c) : container(c) {}
    C& container;
  };


  template <typename D>
  inline
  iterate_<D> iterate(D& d)
  {
    return iterate_<D>(d);
  }

  template <typename F>
  void operator>>(F f, iterate_<const obox2d> it)
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
