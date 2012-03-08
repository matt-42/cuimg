#ifndef CUIMG_PASTE_H_
# define CUIMG_PASTE_H_

# include <cuimg/improved_builtin.h>
# include <cuimg/gpu/image2d.h>
# include <cuimg/error.h>
# include <cuimg/point2d.h>
# include <cuimg/gpu/util.h>

namespace cuimg
{
  namespace internal
  {
    template<typename U, typename V>
    __global__ void paste_kernel(const kernel_image2d<U> in, kernel_image2d<V> out)
    {
      point2d<int> p = thread_pos2d();
      if (in.has(p))
        out(p) = in(p);
    }

    template<typename U, typename V, unsigned US, unsigned VS>
    __global__ void paste_kernel(const kernel_image2d<improved_builtin<U> > in,
                                 kernel_image2d<improved_builtin<V> > out)
    {
      point2d<int> p = thread_pos2d();
      if (in.has(p))
      {
        meta::loop<internal::assign, 0, meta::min_<US, VS>::val - 1>::iter(out(p), in(p));
        meta::loop<internal::reset, meta::min_<US, VS>::val, VS - 1>::iter(out(p));
      }
    }
  }


  template<typename U, typename V>
  void paste(const image2d<U>& in,
             image2d<V>& out,
             dim3 dim_block = dim3(16, 16, 1))
  {
    assert(in.domain() == out.domain());
    dim3 dim_grid = grid_dimension(in.domain(), dim_block);
    internal::paste_kernel<<<dim_grid, dim_block>>>(mki(in), mki(out));
  }

}

#endif
