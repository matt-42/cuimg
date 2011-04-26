#ifndef CUIMG_SIMPLE_PTR_HPP_
# define CUIMG_SIMPLE_PTR_HPP_

namespace cuimg
{

  template <typename T>
  simple_ptr<T>::simple_ptr(const boost::shared_ptr<T>& p)
    : p_(p.get())
  {
  }

  template <typename T>
  simple_ptr<T>::simple_ptr(T* p)
    : p_(p)
  {
  }

  template <typename T>
  simple_ptr<T>::simple_ptr(const simple_ptr<T>& p)
    : p_(p.p_)
  {
  }

  template <typename T>
  simple_ptr<T>&
  simple_ptr<T>::operator=(T* p)
  {
    p_ = p;
    return *this;
  }

  template <typename T>
  simple_ptr<T>&
  simple_ptr<T>::operator=(const simple_ptr<T>& p)
  {
    p_ = p.p_;
    return *this;
  }

  template <typename T>
  T& simple_ptr<T>::operator*()
  {
    return *p_;
  }

  template <typename T>
  T* simple_ptr<T>::operator->()
  {
    return p_;
  }

  template <typename T>
  const T& simple_ptr<T>::operator*() const
  {
    return *p_;
  }

  template <typename T>
  const T* simple_ptr<T>::operator->() const
  {
    return p_;
  }

}

#include <cuimg/simple_ptr.hpp>

#endif
