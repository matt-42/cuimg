#ifndef CUIMG_GAUSSIAN_NOISE
# define CUIMG_GAUSSIAN_NOISE

#include <boost/random/normal_distribution.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>

namespace cuimg
{

  template <typename T, typename G>
  void gaussian_noise(const cuimg::host_image2d<T>& in, cuimg::host_image2d<T>& out,
                      T min, T max, G& generator)
  {
    T middle = (max + min) / 2.f;
    T delta = (max - min) / 2.f;
    for (unsigned r = 0; r < in.nrows(); r++)
      for (unsigned c = 0; c < in.ncols(); c++)
      {
        float p = generator();
        out(r, c) = in(r, c) + (middle + delta * p);
      }
  }

  template <typename T>
  void gaussian_noise(const cuimg::host_image2d<T>& in, cuimg::host_image2d<T>& out, float sigma,
                      T min, T max)
  {
    boost::rand48 engine;

    static int s = time(0);

    engine.seed(s);
    boost::normal_distribution<float> distrib(0.f, sigma);
    boost::variate_generator<boost::rand48, boost::normal_distribution<float> >
      generator(engine, distrib);

    gaussian_noise(in, out,
                   min, max, generator);

    s = engine();
  }

}

#endif
