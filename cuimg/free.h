#ifndef CUIMG_FREE_H_
# define CUIMG_FREE_H_

namespace cuimg
{

  template <typename T>
  void dummy_free(T* p)
  {
  }

  template <typename T>
  void array_free(T* p)
  {
    delete[] p;
  }

}

#endif // !CUIMG_FREE_H_
