#ifndef LAZY_THRUST_ALLOCATOR_H_
# define LAZY_THRUST_ALLOCATOR_H_

#define internal_allocator internal_allocator_old
#include <thrust/detail/internal_allo>
#

namespace cuimg
{

  template<typename T>
  struct my_allocator : thrust::device_malloc_allocator<T>
  {
    typedef thrust::device_malloc_allocator<T> super;

    typedef typename super::pointer   pointer;
    typedef typename super::size_type size_type;

    pointer allocate(size_type n)
    {
      if (blocs_[n].size() != 0)
      {
        p = blocs_[n].top();
        blocs_[n].pop();
        return p;
      }
      else
      {
        pointer pnew;
        cudaMalloc(&pnew, n * sizeof(T));
        return pnew;
      }
    }

    void deallocate(pointer p, size_type n)
    {
      blocs_[n].push(p);
    }

  private:
    static std::map<size_type, std::stack<pointer> > blocs_;
  };

}
#endif
