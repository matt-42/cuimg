#ifndef CUIMG_SIMPLE_PTR_H_
# define CUIMG_SIMPLE_PTR_H_

# include <boost/shared_ptr.hpp>

namespace cuimg
{
  template <typename T>
  class simple_ptr
  {
  public:
    typedef simple_ptr<T> self;

    simple_ptr(const boost::shared_ptr<T>& p);
    simple_ptr(T* p);
    simple_ptr(const self& p);

    self& operator=(T* p);
    self& operator=(const self& p);

    T& operator*();
    T* operator->();

    const T& operator*() const;
    const T* operator->() const;

  private:
    T* p_;
  };

}

#include <cuimg/simple_ptr.hpp>

#endif
