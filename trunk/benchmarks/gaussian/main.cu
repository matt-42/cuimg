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
//#include <cuimg/meta_gaussian_coef.h>
#include <cuimg/meta_gaussian/meta_gaussian_coef_3.h>
#include <cuimg/meta_gaussian/meta_gaussian_coef_100.h>

using namespace cuimg;

#define IMG_SIZE 3072
#define KERNEL_SIZE 80
#define KERNEL_HALF_SIZE_M (KERNEL_SIZE / 2)
#define BLOCKDIM_X (32)
#define BLOCKDIM_Y (1)
#define PIX_PER_THREAD 8
#define APRON_BLOCK 1
#define VTYPE_CUDA float2
#define VTYPE i_float2
#define CONCAT(A, B, C) A ## B ## C
#define XCONCAT(A, B, C) CONCAT(A, B, C)
#define DPOINTS_CPU XCONCAT(c, KERNEL_SIZE ,_rows_cpu)
#define DPOINTS_GPU XCONCAT(c, KERNEL_SIZE ,_rows)
#define ITERATIONS_GPU 10

template <typename T>
struct tex2d;
REGISTER_TEXTURE2D_PROXY(tex2d);


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


texture<float, 1, cudaReadModeElementType> kernel_weights;
texture<int2, 1, cudaReadModeElementType> kernel_dpoints;


template <int R, int E, int N, int SIGMA>
struct gaussian_row_loop
{
  template <typename U>
  static __device__ inline U iter(const kernel_image2d<U>& out, const i_int2& p)
  {;
    return U(tex2D(tex2d<typename U::cuda_bt>::tex(), p.y + R, p.x)) * meta_gaussian_coef<N, SIGMA, R>::coef() +
    gaussian_row_loop<R + 1, E, N, SIGMA>::iter(out, p);
  }
};

template <int E, int N, int SIGMA>
struct gaussian_row_loop<E, E, N, SIGMA>
{
  template <typename U>
  static __device__ inline U iter(const kernel_image2d<U>& out, const i_int2& p)
  {;
    return U(tex2D(tex2d<typename U::cuda_bt>::tex(), p.y, p.x + E)) * meta_gaussian_coef<N, SIGMA, E>::coef();
  }
};

template <typename T, int N, int SIGMA, int KERNEL_HALF_SIZE>
__global__ void gausssian_row_static(kernel_image2d<T> out)
{
  int idr = blockIdx.y * blockDim.y + threadIdx.y;
  int idc = blockIdx.x * blockDim.x + threadIdx.x;
  i_int2 p(idr, idc);
  if (!out.has(p))
    return;
  out(p) = gaussian_row_loop<-KERNEL_HALF_SIZE, KERNEL_HALF_SIZE, N, SIGMA>::iter(out, p);
}


template <typename T, int N>
__global__ void gausssian_row_static_sm(kernel_image2d<T> in, kernel_image2d<T> out)
{
  __shared__ T s_data[BLOCKDIM_X * (PIX_PER_THREAD + APRON_BLOCK * 2)];

  int br = blockIdx.y * blockDim.y;
  int bc = blockIdx.x * PIX_PER_THREAD * blockDim.x;
/*
  int idr = blockIdx.y * blockDim.y + threadIdx.y;
  int idc = blockIdx.x * blockDim.x + threadIdx.x;
  i_int2 p(idr, idc);
  if (!out.has(p))
    return;
*/
  const int minx = bc - (BLOCKDIM_X * APRON_BLOCK) + threadIdx.x;

  T* in_ = (T*)((char*)in.data() + br * in.pitch() + minx * sizeof(T));
  T* out_ = (T*)((char*)out.data() + br * out.pitch() + minx * sizeof(T));

  // main data
  #pragma unroll
  for (int i = APRON_BLOCK; i < PIX_PER_THREAD + APRON_BLOCK; i++)
    s_data[threadIdx.x + i * BLOCKDIM_X] = in_[i * BLOCKDIM_X];

  // left apron
  {
  #pragma unroll
  for (int i = 0;
       i < APRON_BLOCK; i++)
     s_data[threadIdx.x + i * BLOCKDIM_X] = (minx + i * BLOCKDIM_X >= 0) ?
                in_[i * BLOCKDIM_X] : zero();
  }

  {
  // right apron
  #pragma unroll
  for (int i = APRON_BLOCK + PIX_PER_THREAD;
       i < APRON_BLOCK + APRON_BLOCK + PIX_PER_THREAD; i++)
     s_data[threadIdx.x + i * BLOCKDIM_X] =
         (minx  + i * BLOCKDIM_X < in.ncols()) ?
                in_[i * BLOCKDIM_X] : zero();
  }

// convolution
  __syncthreads();
 T res = zero();
 #pragma unroll
 for (int pc = APRON_BLOCK; pc < PIX_PER_THREAD + APRON_BLOCK; pc++)
 {
  #pragma unroll
  for (int i = 0; i < KERNEL_HALF_SIZE_M * 2 + 1; i++)
    res += s_data[pc * BLOCKDIM_X + threadIdx.x + i] * 3.f;
  out_[pc * BLOCKDIM_X] = res;
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
  obox2d domain(IMG_SIZE, IMG_SIZE);
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

  copy(imgh, img);
  copy(img, img_conv_h);

  stats_diff(imgh, img_conv_h);

  print(img);

  float* weights = new float[KERNEL_SIZE];
  for (unsigned i = 0; i < KERNEL_SIZE; i++)
    weights[i] = 1.f;

  float* weights_cuda;
  cudaMalloc(&weights_cuda, KERNEL_SIZE * sizeof(float));
  cudaMemcpy(weights_cuda, weights, KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
   check_cuda_error();
  i_int2* dpoint_cuda;
  cudaMalloc(&dpoint_cuda, KERNEL_SIZE * sizeof(i_int2));
 // cudaMemcpy(dpoint_cuda, DPOINTS_CPU, KERNEL_SIZE * sizeof(i_int2), cudaMemcpyHostToDevice);
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



  float gpu_unrolled_static;
  {
    cudaThreadSynchronize();
    dim3 dimBlock(16, 16);
    dim3 dimGrid(std::ceil(img.ncols() / float(dimBlock.x)), std::ceil(img.nrows() / float(dimBlock.y)));
    reset(img_conv);
    clock_t t = clock();
    for (unsigned i = 0; i < ITERATIONS_GPU; i++)
     // gausssian_row_static<VTYPE, 0, 100, 91><<<dimGrid, dimBlock>>>(kernel_image2d<VTYPE>(img_conv));
      gausssian_row_static<VTYPE, 0, 3, KERNEL_HALF_SIZE_M><<<dimGrid, dimBlock>>>(kernel_image2d<VTYPE>(img_conv));
    check_cuda_error();
    cudaThreadSynchronize();
    check_cuda_error();
    gpu_unrolled_static = 1000*(clock() - t) / float(ITERATIONS_GPU* CLOCKS_PER_SEC);
    std::cout << "row, unrolled, static: " << gpu_unrolled_static << std::endl;
  }

  float gpu_unrolled_static_sm;
  {
    assert( BLOCKDIM_X * APRON_BLOCK >= KERNEL_HALF_SIZE_M );
    assert( img.ncols() % (PIX_PER_THREAD * BLOCKDIM_X) == 0 );

    cudaThreadSynchronize();
    dim3 dimBlock(BLOCKDIM_X, 1);
    dim3 dimGrid(std::ceil(img.ncols() / float(dimBlock.x)) / PIX_PER_THREAD, std::ceil(img.nrows() / float(dimBlock.y)));
    reset(img_conv);
    clock_t t = clock();
//    assert(!(img.ncols() % dimBlock.x));
    for (unsigned i = 0; i < ITERATIONS_GPU; i++)
      gausssian_row_static_sm<VTYPE, 0><<<dimGrid, dimBlock>>>(mki(img), mki(img_conv));
    check_cuda_error();
    cudaThreadSynchronize();
    check_cuda_error();
    gpu_unrolled_static_sm = 1000*(clock() - t) / float(ITERATIONS_GPU* CLOCKS_PER_SEC);
    std::cout << "row, unrolled, static, shared mem: " << gpu_unrolled_static_sm << std::endl;
  }

}
