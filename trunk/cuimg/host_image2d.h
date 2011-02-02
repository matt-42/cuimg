#ifndef CUIMG_HOST_IMAGE2D_H_
# define CUIMG_HOST_IMAGE2D_H_

# include <boost/shared_ptr.hpp>
# include <cuimg/point2d.h>
# include <cuimg/obox2d.h>

namespace cuimg
{
  template <typename V>
  class host_image2d
  {
  public:
    typedef point2d<int> point;
    typedef obox2d<point> domain_type;

    host_image2d(unsigned nrows, unsigned ncols);
    host_image2d(const domain_type& d);

    host_image2d(const host_image2d<V>& d);
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

    V* data();
    const V* data() const;

  private:
    domain_type domain_;
    size_t pitch_;
    boost::shared_ptr<V> data_;
  };

}

# include <cuimg/host_image2d.hpp>

#endif
