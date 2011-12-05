
#include <ctime>
#include <cstdlib>
#include <cuda_runtime.h>

#include <cuimg/improved_builtin.h>
#include <cuimg/cpu/host_image3d.h>
#include <cuimg/gpu/image3d.h>
#include <cuimg/copy.h>

using namespace cuimg;

template <typename T>
void print(const host_image3d<T>& img)
{
  for (unsigned s = 0; s < img.nslices(); s++)
  {
    for (unsigned r = 0; r < img.nrows();   r++)
    {
      for (unsigned c = 0; c < img.ncols();   c++)
        std::cout << img(s,r,c) << std::endl;
      std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << std::endl;
  }
}
int main()
{
  srand(time(0));
  host_image3d<i_float4> img(3, 3, 3);
  host_image3d<i_float4> img_copy(img.domain());
  image3d<i_float4> img_d(img.domain());

  for (unsigned i = 0; i < img.ncols() * img.nrows() * img.nslices(); i++)
    img[i] = i_float4(rand(), rand(), rand(), rand());

  copy(img, img_d);
  copy(img_d, img_copy);

  print(img);
  print(img_copy);

  for (unsigned s = 0; s < img_copy.nslices(); s++)
  for (unsigned r = 0; r < img_copy.nrows();   r++)
  for (unsigned c = 0; c < img_copy.ncols();   c++)
  {
    assert(img(s,r,c) == img_copy(s,r,c));
  }

}
