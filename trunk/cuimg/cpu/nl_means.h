#ifndef CUIMG_NL_MEANS
# define CUIMG_NL_MEANS

namespace cuimg
{

  template <typename T, typename G>
  void nl_means(const cuimg::host_image2d<T>& in, cuimg::host_image2d<T>& out,
                      T min, T max, G& generator)

}

#endif
