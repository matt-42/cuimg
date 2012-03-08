#include <cstdlib>
#include <cassert>
#include <cmath>
#include <iostream>
#include <ctime>
#include <cuda.h>


#include <cuda_runtime.h>

#include <cuimg/improved_builtin.h>
#include <cuimg/builtin_math.h>
#include <cuimg/image2d.h>
#include <cuimg/copy.h>
#include <cuimg/kernel_image2d.h>
#include <cuimg/host_image2d.h>
#include <cuimg/neighb2d_data.h>
#include <cuimg/neighb_iterator2d.h>
#include <cuimg/static_neighb2d.h>
#include <cuimg/texture.h>
#include <cuimg/convolve.h>

using namespace cuimg;

#define IMG_SIZE 3072
#define KERNEL_SIZE 15
#define VTYPE_CUDA float1
#define VTYPE i_float1
#define CONCAT(A, B, C) A ## B ## C
#define XCONCAT(A, B, C) CONCAT(A, B, C)
#define DPOINTS_CPU XCONCAT(c, KERNEL_SIZE ,_rows_cpu)
#define DPOINTS_GPU XCONCAT(c, KERNEL_SIZE ,_rows)
#define ITERATIONS_GPU 100

template <typename T>
struct tex2d;
REGISTER_TEXTURE2D_PROXY(tex2d);


texture<float, 1, cudaReadModeElementType> kernel_weights;
texture<int2, 1, cudaReadModeElementType> kernel_dpoints;

// convolutions 1d:
//   weights: texture, dpoints: texture or static array?

__constant__ const int c3_rows[3][2] = {{-1, 0}, {0, 0}, {1, 0}};
         const int c3_rows_cpu[3][2] = {{-1, 0}, {0, 0}, {1, 0}};

//__constant__ const int c9_rows[9][2] = {{0, -4}, {0, -3}, {0, -2}, {0, -1}, {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}};
//         const int c9_rows_cpu[9][2] = {{0, -4}, {0, -3}, {0, -2}, {0, -1}, {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}};

__constant__ const int c9_rows[9][2] = {{-4, 0}, {-3, 0}, {-2, 0}, {-1, 0}, {0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}};
         const int c9_rows_cpu[9][2] = {{-4, 0}, {-3, 0}, {-2, 0}, {-1, 0}, {0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}};

__constant__ const int c15_rows[15][2] = {{-7, 0}, {-6, 0}, {-5, 0}, {-4, 0}, {-3, 0}, {-2, 0}, {-1, 0}, {0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}};
         const int c15_rows_cpu[15][2] = {{-7, 0}, {-6, 0}, {-5, 0}, {-4, 0}, {-3, 0}, {-2, 0}, {-1, 0}, {0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}};

__constant__ const int c31_rows[31][2] = {{-15, 0}, {-14, 0}, {-13, 0}, {-12, 0}, {-10, 0}, {-9, 0}, {-8, 0}, {-7, 0}, {-6, 0}, {-5, 0}, {-4, 0}, {-3, 0}, {-2, 0},{-1, 0},
                                  {0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}, {9, 0}, {10, 0}, {11, 0}, {12, 0}, {13, 0}, {14, 0}, {15, 0}};
         const int c31_rows_cpu[31][2] = {{-7, 0}, {-6, 0}, {-5, 0}, {-4, 0}, {-3, 0}, {-2, 0}, {-1, 0}, {0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}};


class weigthed_window
{
public:

  weigthed_window(point2d<int>* dpoints,
                  float* weights,
                  unsigned size)
                  : dpoints_(dpoints),
                    weights_(weights),
                    size_(size)
  {
  }

  point2d<int> dpoints(int i)const { return dpoints_[i]; }
  float weights(int i) const { return weights_[i]; }
  unsigned size() const { return size_; }

private:
  point2d<int>* dpoints_;
  float* weights_;
  unsigned size_;
};

template <typename T>
__global__ void convolve_rows_static_global(kernel_image2d<T> out, float* weights)
{
  //int idr = blockIdx.x * blockDim.x + threadIdx.x;
  //int idc = blockIdx.y * blockDim.y + threadIdx.y;
  int idr = blockIdx.y * blockDim.y + threadIdx.y;
  int idc = blockIdx.x * blockDim.x + threadIdx.x;
  i_int2 p(idr, idc);

  if (!out.has(p))
    return;

  bt_change_vtype(T, type_mult(bt_vtype(T), float)) r  = zero();
  neighb_iterator2d<static_neighb2d<KERNEL_SIZE> > n(p, static_neighb2d<KERNEL_SIZE>(DPOINTS_GPU));
  for_all(n) if (out.has(n))
    r += T(tex2D(tex2d<VTYPE_CUDA>::tex(), n->col(), n->row())) * weights[n.i()];
  out(p) = r;
}

template <typename T>
__global__ void convolve_rows_static_texture(kernel_image2d<T> out)
{
  //int idr = blockIdx.x * blockDim.x + threadIdx.x;
  //int idc = blockIdx.y * blockDim.y + threadIdx.y;
  int idr = blockIdx.y * blockDim.y + threadIdx.y;
  int idc = blockIdx.x * blockDim.x + threadIdx.x;
  i_int2 p(idr, idc);

  if (!out.has(p))
    return;

  bt_change_vtype(T, type_mult(bt_vtype(T), float)) r  = zero();
  neighb_iterator2d<static_neighb2d<KERNEL_SIZE> > n(p, static_neighb2d<KERNEL_SIZE>(DPOINTS_GPU));
  for_all(n) if (out.has(n))
  {
    float w = tex1Dfetch(kernel_weights, n.i());
    r += T(tex2D(tex2d<VTYPE_CUDA>::tex(), n->col(), n->row())) * w;
  }
  out(p) = r;
}

template <typename T>
__global__ void convolve_rows_texture_texture(kernel_image2d<T> out, unsigned kernelsize)
{
  //int idr = blockIdx.x * blockDim.x + threadIdx.x;
  //int idc = blockIdx.y * blockDim.y + threadIdx.y;
  int idr = blockIdx.y * blockDim.y + threadIdx.y;
  int idc = blockIdx.x * blockDim.x + threadIdx.x;
  i_int2 p(idr, idc);

  if (!out.has(p))
    return;

  bt_change_vtype(T, type_mult(bt_vtype(T), float)) r  = zero();
  for(int i = 0; i < kernelsize; i++)
  {
    float w = tex1Dfetch(kernel_weights, i);
    point2d<int> n = i_int2(tex1Dfetch(kernel_dpoints, i)) + p;
    if (out.has(n))
      r += T(tex2D(tex2d<VTYPE_CUDA>::tex(), n.col(), n.row())) * w;
  }
  out(p) = r;
}

template <typename T>
__global__ void convolve_rows_loop_texture(kernel_image2d<T> out)
{
  //int idr = blockIdx.x * blockDim.x + threadIdx.x;
  //int idc = blockIdx.y * blockDim.y + threadIdx.y;
  int idr = blockIdx.y * blockDim.y + threadIdx.y;
  int idc = blockIdx.x * blockDim.x + threadIdx.x;
  i_int2 p(idr, idc);

  if (!out.has(p))
    return;

  bt_change_vtype(T, type_mult(bt_vtype(T), float)) r  = zero();
  for(int i = 0; i < KERNEL_SIZE; i++)
  {
    float w = tex1Dfetch(kernel_weights, i);
    point2d<int> n = i_int2(i - KERNEL_SIZE/2, 0) + p;
    if (out.has(n))
      r += T(tex2D(tex2d<VTYPE_CUDA>::tex(), n.col(), n.row())) * w;
  }
  out(p) = r;
}

template <int i>
struct convolve_rows_unrolled_texture_loop
{
  template <typename U, typename T, unsigned N>
  static __device__ void run(const i_int2& p, kernel_image2d<U>& out, improved_builtin<T, N>& r)
  {
    float w = tex1Dfetch(kernel_weights, i);
    point2d<int> n = i_int2(i - KERNEL_SIZE/2, 0) + p;
    //if (out.has(n))
      r += U(tex2D(tex2d<VTYPE_CUDA>::tex(), n.col(), n.row())) * w;
//    int col = p.y + i - KERNEL_SIZE/2;
//    if (col >= 0 && col < out.ncols())
//      r += U(tex2D(tex2d<VTYPE_CUDA>::tex(), col, p.x)) * w;
  }
};


template <int i, int E>
struct convolve_rows_unrolled_texture_loop_inline
{
  template <typename U>
  static __device__ U run(const i_int2& p, kernel_image2d<U>& out)
  {
      return U(tex2D(tex2d<VTYPE_CUDA>::tex(), p.y, p.x + i - KERNEL_SIZE/2)) * 0.132
        + convolve_rows_unrolled_texture_loop_inline<i + 1, E>::run(p, out);
  }
};

template <int E>
struct convolve_rows_unrolled_texture_loop_inline<E, E>
{
  template <typename U>
  static __device__ U run(const i_int2& p, kernel_image2d<U>& out)
  {
      return U(tex2D(tex2d<VTYPE_CUDA>::tex(), p.y, p.x + E - KERNEL_SIZE/2));
  }
};


template <typename T>
__global__ void convolve_rows_unrolled_texture_special(kernel_image2d<T> out)
{
  int idr = blockIdx.y * blockDim.y + threadIdx.y;
  int idc = blockIdx.x * blockDim.x + threadIdx.x;
  i_int2 p(idr, idc);

  if (!out.has(p))
    return;

  bt_change_vtype(T, type_mult(bt_vtype(T), float)) r  = zero();
/*
  convolve_rows_unrolled_texture_loop<31>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<0>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<27>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<1>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<25>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<3>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<23>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<5>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<21>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<7>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<19>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<9>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<17>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<11>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<15>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<16>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<13>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<18>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<14>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<20>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<12>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<22>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<10>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<24>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<8>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<26>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<6>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<28>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<4>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<30>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<2>::run(p, out, r);
  */


  convolve_rows_unrolled_texture_loop<0>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<2>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<1>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<4>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<3>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<6>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<5>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<8>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<7>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<10>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<9>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<12>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<11>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<14>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<13>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<16>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<15>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<18>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<17>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<20>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<19>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<22>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<21>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<24>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<23>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<26>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<25>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<28>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<27>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<30>::run(p, out, r);
  convolve_rows_unrolled_texture_loop<31>::run(p, out, r);

  //meta::special_loop<convolve_rows_unrolled_texture_loop, 0, KERNEL_SIZE - 1>::iter(p, out, r);
  out(p) = r;
}

template <typename T>
__global__ void convolve_rows_unrolled_texture(kernel_image2d<T> out)
{
  //int idr = blockIdx.x * blockDim.x + threadIdx.x;
  //int idc = blockIdx.y * blockDim.y + threadIdx.y;
  int idr = blockIdx.y * blockDim.y + threadIdx.y;
  int idc = blockIdx.x * blockDim.x + threadIdx.x;
  i_int2 p(idr, idc);

  if (!out.has(p))
    return;

  bt_change_vtype(T, type_mult(bt_vtype(T), float)) r  = zero();
//  meta::loop<convolve_rows_unrolled_texture_loop, 0, KERNEL_SIZE - 1>::iter(p, out, r);
  out(p) = convolve_rows_unrolled_texture_loop_inline<0, KERNEL_SIZE - 1>::run(p, out);
}

template <typename T>
void convolve_cpu(host_image2d<T>& in, host_image2d<T>& out, float* weights)
{
  for (unsigned r = 0; r < in.nrows(); r++)
    for (unsigned c = 0; c < in.ncols(); c++)
    {
      point2d<int> p(r, c);
      bt_change_vtype(T, type_mult(bt_vtype(T), float)) res  = zero();
      for(unsigned i = 0; i < KERNEL_SIZE; i++)
      {
        point2d<int> n = i_int2(DPOINTS_CPU[i][0], DPOINTS_CPU[i][1]) + i_int2(p);
        if (in.has(n))
          res += in(n) * weights[i];
      }
      out(p) = res;
    }
}

template <typename T>
void reset(host_image2d<T>& in)
{
  memset(in.data(), 0, in.domain().nrows() * in.domain().ncols() * sizeof(T));
}

template <typename T>
void reset(device_image2d<T>& in)
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
void print(const device_image2d<T>& a)
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
  device_image2d<VTYPE> img(domain);
  device_image2d<VTYPE> img_conv(domain);
  host_image2d<VTYPE> img_conv_h(domain);

  host_image2d<VTYPE> imgh(domain);
  host_image2d<VTYPE> imgh_conv(domain);

  reset(imgh);

  for(unsigned i = 0; i < imgh.nrows(); i++)
    for(unsigned j = 0; j < imgh.ncols(); j++)
      for (unsigned k = 0; k < VTYPE::size; k++)
        imgh(point2d<int>(i, j))[k] = j;

  print(imgh);

//  imgh(point2d<int>(0, 0)).x = 1;
//  imgh(point2d<int>(0, 0)).x = 1;
//  imgh(point2d<int>(100, 100)).x = 1;

  copy(imgh, img);
  copy(img, img_conv_h);

  stats_diff(imgh, img_conv_h);

  print(img);
  float naive_cpu_time,
        gpu_static_global, gpu_static_texture, gpu_texture_texture, gpu_loop_texture, gpu_unroll_texture,gpu_unroll_texture_special, gpu_cuimg;

  float* weights = new float[KERNEL_SIZE];
  for (unsigned i = 0; i < KERNEL_SIZE; i++)
    weights[i] = 1.f;

  {
    clock_t t = clock();
    for (unsigned i = 0; i < 2; i++)
      convolve_cpu(imgh, imgh_conv, weights);

    naive_cpu_time = (clock() - t) / float(2* CLOCKS_PER_SEC);
    std::cout << "naive cpu convolution: " << naive_cpu_time << std::endl;
  }

  print(imgh_conv);


  float* weights_cuda;
  cudaMalloc(&weights_cuda, KERNEL_SIZE * sizeof(float));
  cudaMemcpy(weights_cuda, weights, KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
   check_cuda_error();
  i_int2* dpoint_cuda;
  cudaMalloc(&dpoint_cuda, KERNEL_SIZE * sizeof(i_int2));
  cudaMemcpy(dpoint_cuda, DPOINTS_CPU, KERNEL_SIZE * sizeof(i_int2), cudaMemcpyHostToDevice);
  check_cuda_error();

  // Bind input texture.
  bindTexture2d(img, tex2d<VTYPE_CUDA>::tex());

  { // Bind dpoint texture.
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int2>();
    cudaBindTexture(0, kernel_dpoints, (void*)dpoint_cuda, channelDesc, KERNEL_SIZE * sizeof(i_int2));
    check_cuda_error();
  }
  { // Bind weights texture.
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaBindTexture(0, kernel_weights, (void*)weights_cuda, channelDesc, KERNEL_SIZE * sizeof(float));
    check_cuda_error();
  }

  unsigned d = 16;
  dim3 dimBlock(d, d);
  dim3 dimGrid(std::ceil(img.ncols() / float(d)), std::ceil(img.nrows() / float(d)));

  {
    reset(img_conv);
    clock_t t = clock();
    for (unsigned i = 0; i < 20; i++)
      convolve_rows_static_global<<<dimGrid, dimBlock>>>(kernel_image2d<VTYPE>(img_conv), weights_cuda);
    check_cuda_error();
    cudaThreadSynchronize();
    check_cuda_error();
    gpu_static_global = (clock() - t) / float(20* CLOCKS_PER_SEC);
    std::cout << "dpoints: static, weights: global: " << gpu_static_global << std::endl;
  }
  copy(img_conv, img_conv_h);
  print(img_conv_h);
  stats_diff(imgh_conv, img_conv_h);

  {
    reset(img_conv);
    clock_t t = clock();
    for (unsigned i = 0; i < 20; i++)
      convolve_rows_static_texture<<<dimGrid, dimBlock>>>(kernel_image2d<VTYPE>(img_conv));
    check_cuda_error();
    cudaThreadSynchronize();
    check_cuda_error();
    gpu_static_texture = (clock() - t) / float(20* CLOCKS_PER_SEC);
    std::cout << "dpoints: static, weights: texture: " << gpu_static_texture << std::endl;
  }
  copy(img_conv, img_conv_h);
  print(img_conv_h);
  stats_diff(imgh_conv, img_conv_h);

  {
    reset(img_conv);
    clock_t t = clock();
   for (unsigned i = 0; i < ITERATIONS_GPU; i++)
     convolve_rows_texture_texture<<<dimGrid, dimBlock>>>(kernel_image2d<VTYPE>(img_conv), KERNEL_SIZE);
    check_cuda_error();
    cudaThreadSynchronize();
    gpu_texture_texture = (clock() - t) / float(ITERATIONS_GPU* CLOCKS_PER_SEC);
    std::cout << "dpoints: texture, weights: texture: " << gpu_texture_texture << std::endl;
  }
  copy(img_conv, img_conv_h);
  print(img_conv_h);
  stats_diff(imgh_conv, img_conv_h);

  {
    reset(img_conv);
    clock_t t = clock();
   for (unsigned i = 0; i < ITERATIONS_GPU; i++)
     convolve_rows_loop_texture<<<dimGrid, dimBlock>>>(kernel_image2d<VTYPE>(img_conv));
    check_cuda_error();
    cudaThreadSynchronize();
    gpu_loop_texture = (clock() - t) / float(ITERATIONS_GPU* CLOCKS_PER_SEC);
    std::cout << "dpoints: loop, weights: texture: " << gpu_loop_texture << std::endl;
  }

  copy(img_conv, img_conv_h);
  print(img_conv_h);
  stats_diff(imgh_conv, img_conv_h);

  {
    reset(img_conv);
    clock_t t = clock();
    for (unsigned i = 0; i < ITERATIONS_GPU; i++)
     convolve_rows_unrolled_texture<<<dimGrid, dimBlock>>>(kernel_image2d<VTYPE>(img_conv));
    check_cuda_error();
    cudaThreadSynchronize();
    gpu_unroll_texture = (clock() - t) / float(ITERATIONS_GPU* CLOCKS_PER_SEC);
    std::cout << "dpoints: unrolled loop, weights: texture: " << gpu_unroll_texture << std::endl;
  }

  copy(img_conv, img_conv_h);
  print(img_conv_h);
  stats_diff(imgh_conv, img_conv_h);

  {
    reset(img_conv);
    clock_t t = clock();
    for (unsigned i = 0; i < ITERATIONS_GPU; i++)
     convolve_rows_unrolled_texture_special<<<dimGrid, dimBlock>>>(kernel_image2d<VTYPE>(img_conv));
    check_cuda_error();
    cudaThreadSynchronize();
    gpu_unroll_texture_special = (clock() - t) / float(ITERATIONS_GPU* CLOCKS_PER_SEC);
    std::cout << "dpoints: unrolled loop special, weights: texture: " << gpu_unroll_texture_special << std::endl;
  }

  copy(img_conv, img_conv_h);
  print(img_conv_h);
  stats_diff(imgh_conv, img_conv_h);

  {
    reset(img_conv);
    clock_t t = clock();
    weigthed_window ww((point2d<int>*)DPOINTS_CPU, weights, KERNEL_SIZE);
    for (unsigned i = 0; i < ITERATIONS_GPU; i++)
     convolve(img, img_conv, ww);
    check_cuda_error();
    cudaThreadSynchronize();
    gpu_cuimg = (clock() - t) / float(ITERATIONS_GPU* CLOCKS_PER_SEC);
    std::cout << "cuimg: " << gpu_cuimg << std::endl;
  }

  copy(img_conv, img_conv_h);
  print(img_conv_h);
  stats_diff(imgh_conv, img_conv_h);


  std::cout << "Speed up: " << std::endl;
  std::cout << "dpoints: static, weights: global : " << (naive_cpu_time / gpu_static_global) << " x" << std::endl;
  std::cout << "dpoints: static, weights: texture: " << (naive_cpu_time / gpu_static_texture) << " x" << std::endl;
  std::cout << "dpoints: texture, weights: texture: " << (naive_cpu_time / gpu_texture_texture) << " x" << std::endl;
  std::cout << "dpoints: loop, weights: texture: " << (naive_cpu_time / gpu_loop_texture) << " x" << std::endl;
  std::cout << "dpoints: unroll, weights: texture: " << (naive_cpu_time / gpu_unroll_texture) << " x" << std::endl;
  std::cout << "dpoints: unroll special , weights: texture: " << (naive_cpu_time / gpu_unroll_texture_special) << " x" << std::endl;
  std::cout << "cuimg: " << (naive_cpu_time / gpu_cuimg) << " x" << std::endl;
}
