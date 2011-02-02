#include <cstdlib>
#include <cassert>
#include <cmath>
#include <iostream>
#include <ctime>
#include <cuda.h>


#include <cuda_runtime.h>

#include <cuimg/improved_builtin.h>
#include <cuimg/image2d.h>
#include <cuimg/image2d_math.h>
#include <cuimg/copy.h>
#include <cuimg/kernel_image2d.h>
#include <cuimg/host_image2d.h>
#include <cuimg/neighb2d_data.h>
#include <cuimg/neighb_iterator2d.h>
#include <cuimg/static_neighb2d.h>

//#include <cv.h>
//#include <highgui.h>


using namespace cuimg;

#define IMG_SIZE 3072
#define KERNEL_HALF_SIZE 1
#define KERNEL_SIZE ((KERNEL_HALF_SIZE * 2 + 1) * (KERNEL_HALF_SIZE * 2 + 1))
#define KERNEL_DPOINTS c9
#define VTYPE_CUDA float1
#define VTYPE i_float1
#define GPU_ITERATIONS 50

texture<VTYPE_CUDA, 2, cudaReadModeElementType> tex;


template <typename T>
__global__ void convolve(kernel_image2d<T> in, kernel_image2d<T> out)
{
  int idr = blockIdx.y * blockDim.y + threadIdx.y;
  int idc = blockIdx.x * blockDim.x + threadIdx.x;
  i_int2 p(idr, idc);

  if (!in.has(p))
    return;

  T sum = zero();
  unsigned w = 0;
  for (int i = -KERNEL_HALF_SIZE ; i <= KERNEL_HALF_SIZE ; i++)
  for (int j = -KERNEL_HALF_SIZE ; j <= KERNEL_HALF_SIZE ; j++)
  {
    i_int2 n = p + i_int2(i, j);
    if (in.has(n))
    {
      sum += in(n);
      w++;
    }
  }
  out(p) = sum / w;
}


template <typename T>
__global__ void convolve_tex(kernel_image2d<T> out)
{
  int idr = blockIdx.y * blockDim.y + threadIdx.y;
  int idc = blockIdx.x * blockDim.x + threadIdx.x;
  i_int2 p(idr, idc);

  if (!out.has(p))
    return;

  T sum = zero();
  unsigned w = 0;
  for (int i = -KERNEL_HALF_SIZE ; i <= KERNEL_HALF_SIZE ; i++)
  for (int j = -KERNEL_HALF_SIZE ; j <= KERNEL_HALF_SIZE ; j++)
  {
    i_int2 n = p + i_int2(i, j);
    if (out.has(n))
    {
      sum += T(tex2D(tex, n.y, n.x));
      w++;
    }
  }
  out(p) = sum / w;
}

template <int R, int C, int E>
struct conv_c_loop
{
  template <typename U, typename T>
  static __device__ inline void iter(const kernel_image2d<U>& out, const i_int2& p, T& sum, unsigned& w)
  {
    i_int2 n = p + i_int2(R, C);
    if (out.has(n))
    {
      sum += T(tex2D(tex, n.y, n.x));
      w++;
    }
    conv_c_loop<R, C + 1, E>::iter(out, p, sum, w);
  }
};

template <int R, int E>
struct conv_c_loop<R, E, E>
{
  template <typename U, typename T>
  static __device__ inline  void iter(const kernel_image2d<U>& out, const i_int2& p, T& sum, unsigned& w)
  {
    i_int2 n = p + i_int2(R, E);
    if (out.has(n))
    {
      sum += T(tex2D(tex, float(n.y), float(n.x)));
      w++;
    }
  }
};

template <int R, int E>
struct conv_r_loop
{
  template <typename U, typename T>
  static __device__ void iter(const kernel_image2d<U>& out, const i_int2& p, T& sum, unsigned& w)
  {
    conv_c_loop<R, -KERNEL_HALF_SIZE, KERNEL_HALF_SIZE>::iter(out, p, sum, w);
    conv_r_loop<R + 1, KERNEL_HALF_SIZE>::iter(out, p, sum, w);
  }
};

template <int E>
struct conv_r_loop<E, E>
{
  template <typename U, typename T>
  static __device__ void iter(const kernel_image2d<U>& out, const i_int2& p, T& sum, unsigned& w)
  {
    conv_c_loop<E, -KERNEL_HALF_SIZE, KERNEL_HALF_SIZE>::iter(out, p, sum, w);
  }
};

template <typename T>
__global__ void convolve_tex_unrolled(kernel_image2d<T> out)
{
  int idr = blockIdx.y * blockDim.y + threadIdx.y;
  int idc = blockIdx.x * blockDim.x + threadIdx.x;
  i_int2 p(idr, idc);

  if (!out.has(p))
    return;

  T sum = zero();
  unsigned w = 0;
  conv_r_loop<-KERNEL_HALF_SIZE, KERNEL_HALF_SIZE>::iter(out, p, sum, w);

  out(p) = sum / w;
}


template <typename T>
__global__ void convolve_tex_static(kernel_image2d<T> out)
{
  int idr = blockIdx.y * blockDim.y + threadIdx.y;
  int idc = blockIdx.x * blockDim.x + threadIdx.x;
  i_int2 p(idr, idc);

  if (!out.has(p))
    return;

  T sum = zero();
  unsigned w = 0;
  static_neighb2d<KERNEL_SIZE> nn(KERNEL_DPOINTS);
  for (int i = 0; i < KERNEL_SIZE; i++)
  {
    i_int2 n = p + i_int2(nn[i]);
    if (out.has(n))
    {
      sum += T(tex2D(tex, n.y, n.x));
      w++;
    }
  }
  out(p) = sum / w;
}


template <typename T>
__global__ void convolve_tex_static_it(kernel_image2d<T> out)
{
  int idr = blockIdx.y * blockDim.y + threadIdx.y;
  int idc = blockIdx.x * blockDim.x + threadIdx.x;
  i_int2 p(idr, idc);

  if (!out.has(p))
    return;

  T sum = zero();
  unsigned w = 0;
  neighb_iterator2d<static_neighb2d<KERNEL_SIZE> > n(p, static_neighb2d<KERNEL_SIZE>(KERNEL_DPOINTS));
  for(n.start(); n.is_valid(); n.next() )
  {
    if (out.has(n))
    {
      sum += T(tex2D(tex, n->col(), n->row()));
      w++;
    }
  }
  out(p) = sum / w;
}



template <typename T>
void convolve_cpu(host_image2d<T>& in, host_image2d<T>& out)
{

  for (unsigned r = 0; r < in.nrows(); r++)
    for (unsigned c = 0; c < in.ncols(); c++)
    {
      point2d<int> p(r, c);

      T sum = zero();
      unsigned w = 0;
      for (int i = -KERNEL_HALF_SIZE ; i <= KERNEL_HALF_SIZE ; i++)
        for (int j = -KERNEL_HALF_SIZE ; j <= KERNEL_HALF_SIZE ; j++)
        {
          i_int2 n = i_int2(p) + i_int2(i, j);
          if (in.has(n))
          {
            sum += in(n);
            w++;
          }
        }
        out(p) = sum / w;
    }

}

template <typename T>
void reset(host_image2d<T>& in)
{
  memset(in.data(), 0, in.domain().nrows() * in.domain().ncols() * sizeof(T));
}

template <typename T>
void reset(image2d<T>& in)
{
  cudaMemset(in.data(), 0, in.domain().nrows() * in.pitch());
}

template <typename T>
float diff(host_image2d<T>& a, host_image2d<T>& b)
{
  float res = 0;
  for(unsigned i = 0; i < a.nrows(); i++)
    for(unsigned j = 0; j < a.ncols(); j++)
      res += norml2(a(i, j) - b(i, j));
  return res;
}

template <typename T>
void stats_diff(host_image2d<T>& a, host_image2d<T>& b)
{
  float max = 0;
  float sum = 0;
  for(unsigned i = 0; i < a.nrows(); i++)
    for(unsigned j = 0; j < a.ncols(); j++)
    {
      float d = norml2(a(i, j) - b(i, j));
      sum += d;
      if (d > max) max = d;
    }

  std::cout << "max diff per pixel: " << max << std::endl;
  std::cout << "sum: " << sum << std::endl;
  std::cout << "mean: " << sum / (a.nrows() * a.ncols()) << std::endl;
}

template <typename T>
void print(const host_image2d<T>& a)
{
  if (a.nrows() * a.ncols() > 20)
    return;

  std::cout << "-----------------------" << std::endl;
  for(unsigned i = 0; i < a.nrows(); i++)
  {
    for(unsigned j = 0; j < a.ncols(); j++)
      std::cout << a(i, j) << ",\t";
    std::cout << std::endl;
  }
  std::cout << "-----------------------" << std::endl;
}

template <typename T>
void print(const image2d<T>& a)
{
  if (a.nrows() * a.ncols() > 20)
    return;
  host_image2d<T> tmp(a.domain());
  copy(a, tmp);
  print(tmp);
}


int main()
{
  srand(time(0));
  obox2d<point2d<int> > domain(IMG_SIZE, IMG_SIZE);
  image2d<VTYPE> img(domain);
  image2d<VTYPE> img_conv(domain);
  host_image2d<VTYPE> img_conv_h(domain);

  host_image2d<VTYPE> imgh(domain);
  host_image2d<VTYPE> imgh_conv(domain);

  reset(imgh);

  for(unsigned i = 0; i < imgh.nrows(); i++)
    for(unsigned j = 0; j < imgh.ncols(); j++)
      for (unsigned k = 0; k < VTYPE::size; k++)
        imgh(point2d<int>(i, j))[k] = i;

  print(imgh);

//  imgh(point2d<int>(0, 0)).x = 1;
//  imgh(point2d<int>(0, 0)).x = 1;
//  imgh(point2d<int>(100, 100)).x = 1;

  copy(imgh, img);
  copy(img, img_conv_h);
  
  stats_diff(imgh, img_conv_h);

  print(img);
  float naive_cpu_time,
        texture_gpu_time, texture_unroll_gpu_time, texture_static_gpu_time,
        texture_static_it_gpu_time;

  {
    clock_t t = clock();
    for (unsigned i = 0; i < 1; i++)
      convolve_cpu(imgh, imgh_conv);
  
     naive_cpu_time = (clock() - t) / float(1* CLOCKS_PER_SEC);
     std::cout << "naive cpu convolution: " << naive_cpu_time << std::endl;
  }

  print(imgh_conv);

  std::cout << imgh_conv(point2d<int>(0, 0)).x << std::endl;

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VTYPE_CUDA>();;
  cudaBindTexture2D(0, tex, (void*)img.data(), channelDesc, img.ncols(), img.nrows(), img.pitch());
  check_cuda_error();

  unsigned d = 16;
  dim3 dimBlock(d, d);
  dim3 dimGrid(idivup(img.ncols(), d), idivup(img.nrows(), d));

   {
     reset(img_conv);
    clock_t t = clock();
    for (unsigned i = 0; i < GPU_ITERATIONS; i++)
      convolve_tex_unrolled<<<dimGrid, dimBlock>>>(kernel_image2d<VTYPE>(img_conv));
    cudaThreadSynchronize();
    texture_unroll_gpu_time = (clock() - t) / float(GPU_ITERATIONS* CLOCKS_PER_SEC);
    std::cout << "texture unrolled gpu convolution: " << texture_unroll_gpu_time << std::endl;
  }

  copy(img_conv, img_conv_h);
  print(img_conv_h);
  stats_diff(imgh_conv, img_conv_h);

  {
    reset(img_conv);
    clock_t t = clock();
    for (unsigned i = 0; i < GPU_ITERATIONS; i++)
      convolve_tex_static<<<dimGrid, dimBlock>>>(kernel_image2d<VTYPE>(img_conv));
    cudaThreadSynchronize();
    texture_static_gpu_time = (clock() - t) / float(GPU_ITERATIONS* CLOCKS_PER_SEC);
    std::cout << "texture static gpu convolution: " << texture_static_gpu_time << std::endl;
  }

  copy(img_conv, img_conv_h);
  print(img_conv_h);
  stats_diff(imgh_conv, img_conv_h);

  {
    reset(img_conv);
    clock_t t = clock();
    for (unsigned i = 0; i < GPU_ITERATIONS; i++)
      convolve_tex_static_it<<<dimGrid, dimBlock>>>(kernel_image2d<VTYPE>(img_conv));
    cudaThreadSynchronize();
    texture_static_it_gpu_time = (clock() - t) / float(GPU_ITERATIONS* CLOCKS_PER_SEC);
    std::cout << "texture static  it gpu convolution: " << texture_static_it_gpu_time << std::endl;
  }

  copy(img_conv, img_conv_h);
  print(img_conv_h);
  stats_diff(imgh_conv, img_conv_h);

    {
    reset(img_conv);
    clock_t t = clock();
    for (unsigned i = 0; i < GPU_ITERATIONS; i++)
      convolve_tex<<<dimGrid, dimBlock>>>(kernel_image2d<VTYPE>(img_conv));
    cudaThreadSynchronize();
    texture_gpu_time = (clock() - t) / float(GPU_ITERATIONS* CLOCKS_PER_SEC);
    std::cout << "texture gpu convolution: " << texture_gpu_time << std::endl;
    check_cuda_error();
  }

  copy(img_conv, img_conv_h);
  print(img_conv_h);
  stats_diff(imgh_conv, img_conv_h);

  std::cout << "Speed up: " << std::endl;
  std::cout << "texture gpu convolution: " << (naive_cpu_time / texture_gpu_time) << " x" << std::endl;
  std::cout << "texture unrolled gpu convolution: " << (naive_cpu_time / texture_unroll_gpu_time) << " x" << std::endl;
  std::cout << "texture static gpu convolution: " << (naive_cpu_time / texture_static_gpu_time) << " x" << std::endl;
  std::cout << "texture static it gpu convolution: " << (naive_cpu_time / texture_static_it_gpu_time) << " x" << std::endl;

}
