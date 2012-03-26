#ifndef CUIMG_HOST_IMAGE2D_H_
# define CUIMG_HOST_IMAGE2D_H_

# include <boost/shared_ptr.hpp>
# include <omp.h>

# include <cuimg/target.h>
# include <cuimg/concepts.h>
# include <cuimg/dsl/expr.h>
# include <cuimg/point2d.h>
# include <cuimg/obox2d.h>

#ifdef WITH_OPENCV
# include <opencv/cxcore.h>
#endif


namespace cuimg
{
  template <typename V>
  class host_image2d : public Image2d<host_image2d<V> >
  {
  public:
    enum { target = CPU };

    typedef V value_type;
    typedef point2d<int> point;
    typedef obox2d<point> domain_type;
    typedef boost::shared_ptr<V> PT;

    host_image2d();
    host_image2d(unsigned nrows, unsigned ncols, bool pinned = 0);
    host_image2d(const domain_type& d, bool pinned = 0);

    host_image2d(const host_image2d<V>& d);
#ifdef WITH_OPENCV
    host_image2d(IplImage * imgIpl);
#endif
    host_image2d<V>& operator=(const host_image2d<V>& d);

    const domain_type& domain() const;
    unsigned nrows() const;
    unsigned ncols() const;
    bool has(const point& p) const;

    size_t pitch() const;

    V& operator()(const point& p);
    const V& operator()(const point& p) const;

    V& operator()(int r, int c);
    const V& operator()(int r, int c) const;

#ifdef WITH_OPENCV
    IplImage* getIplImage();
    host_image2d<V>& operator=(IplImage * imgIpl);
#endif

    V* row(unsigned i);
    const V* row(unsigned i) const;

    V* data();
    const V* data() const;

    size_t buffer_size() const;

    template <typename E>
    host_image2d<V>& operator=(const expr<E>& e)
    {
      const E& x(*(E*)&e);
#pragma omp parallel for schedule(static, 8)
      for (unsigned r = 0; r < nrows(); r++)
        for (unsigned c = 0; c < ncols(); c++)
          (*this)(r, c) = x.eval(point2d<int>(r, c));
      return *this;
    }

  private:
    void allocate(const domain_type& d, bool pinned);

    domain_type domain_;
    size_t pitch_;
    boost::shared_ptr<V> data_;
    V* buffer_;
  };

}
# include <cuimg/cpu/host_image2d.hpp>

#endif
