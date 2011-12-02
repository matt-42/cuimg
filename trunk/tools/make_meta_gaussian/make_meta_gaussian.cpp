#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

#define CUIMG_PI 3.14159265f

template <int N>
float gaussian_derivative(float sigma, int x);

template <>
float gaussian_derivative<0>(float sigma, int x)
{
  return (1 / (sqrt(2 * CUIMG_PI) * sigma))
    * exp(-(x * x) / (2 * sigma * sigma));
}

template <>
float gaussian_derivative<1>(float sigma, int x)
{
  return -(x / (sqrt(2 * CUIMG_PI) * std::pow(sigma, 3)))
    * exp(-(x * x) / (2 * sigma * sigma));
}

template <>
float gaussian_derivative<2>(float sigma, int x)
{
  return (((x - sigma) * (x + sigma)) /
    (sqrt(2 * CUIMG_PI) * std::pow(sigma, 5)))
    * exp(-(x * x) / (2 * sigma * sigma));
}

template <>
float gaussian_derivative<3>(float sigma, int x)
{
  return ((x * (x * x - 3 * sigma * sigma)) /
    (sqrt(2 * CUIMG_PI) * std::pow(sigma, 7)))
    * exp(-(x * x) / (2 * sigma * sigma));
}

template <>
float gaussian_derivative<4>(float sigma, int x)
{
  return ((x*x*x*x - 6 * x*x * sigma * sigma + 3 * std::pow(sigma, 4)) /
    (sqrt(2 * CUIMG_PI) * std::pow(sigma, 9)))
    * exp(-(x * x) / (2 * sigma * sigma));
}

float gaussian_derivative(int n, float sigma, int x)
{
  switch(n)
  {
  case 0: return gaussian_derivative<0>(sigma, x);
  case 1: return gaussian_derivative<1>(sigma, x);
  case 2: return gaussian_derivative<2>(sigma, x);
  case 3: return gaussian_derivative<3>(sigma, x);
  case 4: return gaussian_derivative<4>(sigma, x);
  default: return 0;
  }
}


void make_sigma_specific_file(const std::string& path, int sigma)
{
  float x = -7.6946e-022f;
  std::stringstream filename_ss; filename_ss << path << "/meta_gaussian_coefs_" << sigma << ".h";

  std::ofstream of(filename_ss.str());
  of << "// This file is generated." << std::endl;
  of << "#ifndef CUIMG_META_GAUSSIAN_" << sigma << "_H_" << std::endl;
  of << "# define CUIMG_META_GAUSSIAN_" << sigma << "_H_" << std::endl;
  of << std::endl;
  of << std::endl;
  of << "#include \"meta_gaussian.h\""<<  std::endl;
  of << std::endl;
  of << std::endl;
  
  of << "namespace cuimg" << std::endl;
  of << "{" << std::endl;
  of << "// sigma = " << sigma << "." << std::endl;
  for (unsigned n = 0; n <= 4; n++)
  {
    of << "// " << n << " th derivative." << std::endl;
    for (int x = -50; x <= 50; x++)
    {
      float res = gaussian_derivative(n, float(sigma), x);
      if (res != 0.f)
      {
        std::stringstream ss; ss << res;
        if (ss.str().find('.') != std::string::npos)
          ss << "f";
        else
          ss << ".f";

        of << "template <> struct meta_kernel<meta_gaussian<" << n << ", " << sigma << ">, " << x
          << "> { static __host__ __device__ float coef() { return "
          << ss.str() << "; } }; " << std::endl;
      of << std::endl;
      }
    }
  }
  
  of << "}" << std::endl;
  of << "#endif //! CUIMG_META_GAUSSIAN_" << sigma << "_H_" << std::endl;
}

void make_all_sigma_file(const std::string& path)
{
  
  std::stringstream filename_ss; filename_ss << path << "/meta_gaussian_coefs.h";
  std::ofstream of(filename_ss.str());

  of << "// This file is generated." << std::endl;
  of << "#ifndef CUIMG_META_GAUSSIAN_H_" << std::endl;
  of << "# define CUIMG_META_GAUSSIAN_H_" << std::endl;
  of << std::endl;
  of << std::endl;
  of << "#include \"meta_gaussian.h\""<<  std::endl;
  of << std::endl;
  of << std::endl;

  of << "namespace cuimg" << std::endl;
  of << "{" << std::endl;

  for (unsigned n = 0; n <= 2; n++)
  {
    of << "// " << n << " th derivative." << std::endl;
    for (unsigned s = 1; s <= 10; s++)
    {
      of << "// sigma = " << s << "." << std::endl;
      for (int x = -50; x <= 50; x++)
      {
        float res = gaussian_derivative(n, float(s), x);
        if (res != 0.f)
        {
          std::stringstream ss; ss << res;
          if (ss.str().find('.') != std::string::npos)
            ss << "f";
          else
            ss << ".f";

          of << "template <> struct meta_kernel<meta_gaussian<" << n << ", " << s << ">, " << x
             << "> { static __host__ __device__ float coef() { return "
             << res << "; } }; " << std::endl;
        }
 }
      of << std::endl;
    }
  }
  of << "}" << std::endl;

}

int usage()
{
   std::cout << "Usage: ./make_meta_gaussian [-g for one file|-s to split coeficients in sigma specific files] path" << std::endl;
   return 1;
}

int main(int argc, char* argv[])
{
  if (argc != 3) return usage();

  if (std::string(argv[1]) == "-g")
    make_all_sigma_file(argv[2]);
  else if (std::string(argv[1]) == "-s")
  {
    for (unsigned s = 1; s <= 50; s++)
      make_sigma_specific_file(argv[2], s);
  }
  else
    return usage();

  return 0;
}
