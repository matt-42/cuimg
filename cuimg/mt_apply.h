#ifndef CUIMG_OMP_APPLY_H_
# define CUIMG_OMP_APPLY_H_

# include <cuimg/architectures.h>

namespace cuimg
{

  template <typename ARCH, typename F>
  inline void mt_apply2d(int elt_size, const obox2d& domain, const F& f, ARCH = arch::cpu())
  {
    dim3 dimblock = ::cuimg::dimblock<ARCH>(elt_size, domain);
#pragma omp parallel for schedule(static, dimblock.y)
    for (unsigned r = 0; r < domain.nrows(); r++)
      for (unsigned c = 0; c < domain.ncols(); c++)
	f(i_int2(r, c));
  }

}

#endif
