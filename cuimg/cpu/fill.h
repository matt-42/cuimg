#ifndef CUIMG_CPU_FILL_H_
# define CUIMG_CPU_FILL_H_

# include <cuimg/improved_builtin.h>
# include <cuimg/cpu/host_image2d.h>
# include <cuimg/error.h>
# include <cuimg/util.h>

namespace cuimg
{

  template<typename U>
  void fill(host_image2d<U>& out,
            const U& v)
  {
    unsigned nr = out.nrows();
    unsigned nc = out.ncols();
#pragma omp parallel for schedule(static, 2)
    for (unsigned r = 0; r < nr; r++)
    {
      U* row = &(out(r, 0));
      U* row_end = &(out(r, nc - 1));
      while (row <= row_end)
      {
	*row = v;
	++row;
      }
    }
  }

}

#endif
