#ifndef CUIMG_HOST_IMAGE3D_H_
# define CUIMG_HOST_IMAGE3D_H_

# include <boost/shared_ptr.hpp>
# include <cuimg/point3d.h>
# include <cuimg/obox3d.h>

namespace cuimg
{
  template <typename V>
  class host_image3d
  {
  public:
    typedef point3d<int> point;
    typedef obox3d<point> domain_type;

    host_image3d(unsigned nslices, unsigned nrows, unsigned ncols);
    host_image3d(const domain_type& d);

    host_image3d(const host_image3d<V>& d);
    host_image3d<V>& operator=(const host_image3d<V>& d);

    const domain_type& domain() const;
    unsigned nslices() const;
    unsigned nrows() const;
    unsigned ncols() const;
    bool has(const point& p) const;

    size_t pitch() const;

    V& operator()(const point& p);
    const V& operator()(const point& p) const;

    V& operator()(int s, int r, int c);
    const V& operator()(int s, int r, int c) const;

    V& operator[](unsigned i);
    const V& operator[](unsigned i) const;

    V* data();
    const V* data() const;

  private:
    domain_type domain_;
    size_t pitch_;
    boost::shared_ptr<V> data_;
  };

}

# include <cuimg/host_image3d.hpp>

#endif
