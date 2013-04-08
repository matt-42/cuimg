#ifndef CUIMG_GAUSSIAN_NOISE
# define CUIMG_GAUSSIAN_NOISE

#include <limits>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>

#include <cuimg/cpu/host_image2d.h>

namespace cuimg
{

#ifndef NO_CPP0X
  template <typename T, typename U, typename G>
  void gaussian_noise(const cuimg::host_image2d<T>& in, cuimg::host_image2d<U>& out,
                      G& generator)
  {
    typedef typename T::vtype vtype;
    vtype max = std::numeric_limits<vtype>::max();
    vtype min = std::numeric_limits<vtype>::min();
    for (unsigned r = 0; r < in.nrows(); r++)
      for (unsigned c = 0; c < in.ncols(); c++)
      {
        float p = generator();
	T v = in(r, c);
	v.for_each_comp([&generator, min, max] (vtype& c)
			{
			  auto res = c + generator();
			  if (res > max) res = max;
			  else if (res < min) res = min;
			  c = res;
			});
	out(r, c) = v;
      }
  }

#endif

  template <typename T, typename U>
  void gaussian_noise(const cuimg::host_image2d<T>& in, cuimg::host_image2d<U>& out, float sigma,
                      unsigned seed = 0)
  {
    boost::rand48 engine;

    static int s = time(0);
    if (!seed)
    {
      engine.seed(s);
    }
    else
      engine.seed(seed);
    boost::normal_distribution<float> distrib(0.f, sigma);
    boost::variate_generator<boost::rand48, boost::normal_distribution<float> >
      generator(engine, distrib);

    gaussian_noise(in, out, generator);

    if (!seed)
      s = engine();
  }

  template <typename U>
  void gaussian_noise(cuimg::host_image2d<U>& out, float sigma,
                      unsigned seed = 0)
  {
    gaussian_noise(out, out, sigma, seed);
  }

}

#endif
