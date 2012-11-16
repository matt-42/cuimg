#ifndef CUIMG_OMP_APPLY_H_
# define CUIMG_OMP_APPLY_H_

# include <cuimg/architectures.h>
# include <cuimg/obox2d.h>
# include <cuimg/box2d.h>

namespace cuimg
{

  template <typename ARCH, typename F>
  inline void mt_apply2d(int elt_size, const obox2d& domain, const F& f, ARCH = arch::cpu())
  {
    dim3 dimblock = ::cuimg::dimblock(ARCH(), elt_size, domain);
#pragma omp parallel for schedule(static, dimblock.y)
    for (unsigned r = 0; r < domain.nrows(); r++)
      for (unsigned c = 0; c < domain.ncols(); c++)
        f(i_int2(r, c));
  }


  template <typename ARCH, typename F>
  inline void mt_apply2d(int elt_size, const box2d& domain, const F& f, ARCH = arch::cpu())
  {
    dim3 dimblock = ::cuimg::dimblock(ARCH(), elt_size, domain);
#pragma omp parallel for schedule(static, dimblock.y)
    for (int r = domain.p1().r(); r <= domain.p2().r(); r++)
      for (int c = domain.p1().c(); c <= domain.p2().c(); c++)
        f(i_int2(r, c));
  }

  

  template <typename ARCH, typename F>
  inline void st_apply2d(int elt_size, const box2d& domain, const F& f, ARCH = arch::cpu())
  {
    for (int r = domain.p1().r(); r <= domain.p2().r(); r++)
      for (int c = domain.p1().c(); c <= domain.p2().c(); c++)
        f(i_int2(r, c));
  }


  template <typename F, typename D>
  inline void mt_apply2d(const D& domain, const F& f)
  {
    mt_apply2d(1, domain, f, arch::cpu());
  }

}

#endif
