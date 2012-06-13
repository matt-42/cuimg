
#include <iostream>
#include <ctime>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cuimg/improved_builtin.h>
#include <cuimg/error.h>

#define BUFFER_SIZE (1000*100)

#define N (1000*100)

using namespace cuimg;

__global__ void wo_cuimg(float4* buffer)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id >= BUFFER_SIZE)
    return;

  float4 f = buffer[id];
  for(int i = 0; i < N; i++)
  {
    f.x = f.x + f.x;
    f.y = f.y + f.y;
    f.z = f.z + f.z;
    f.w = f.w + f.w;
  }
  buffer[id].x = f.x + 42;
  buffer[id].y = f.y + 43;
  buffer[id].z = f.z + 44;
  buffer[id].w = f.w + 45;
}

__global__ void w_cuimg(i_float4* buffer)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id >= BUFFER_SIZE)
    return;

  i_float4 f = buffer[id];
  for(int i = 0; i < N; i++)
    f = f + f;
  buffer[id] = f + i_float4(42, 43, 44, 45);
}

int main()
{
  float4* buffer_cpu = new float4[BUFFER_SIZE]();
  float4* buffer;
  cudaMalloc(&buffer, BUFFER_SIZE * sizeof(float4));
  cuimg::check_cuda_error();
  assert(buffer);
  cudaMemset(buffer, 0, BUFFER_SIZE * sizeof(float4));
  cuimg::check_cuda_error();

  unsigned d = 16;
  dim3 dimBlock(d);
  dim3 dimGrid(std::ceil(BUFFER_SIZE / float(d)));

  size_t time = std::clock();
  wo_cuimg<<<dimGrid, dimBlock>>>(buffer);
  cuimg::check_cuda_error();
  cudaMemcpy(buffer_cpu, buffer, sizeof(float4), cudaMemcpyDeviceToHost);
  cuimg::check_cuda_error();
  std::cout << "pure cuda: "<< (float(clock() - time) / float(CLOCKS_PER_SEC)) << std::endl;

  cudaMemset(buffer, 0, BUFFER_SIZE * sizeof(float4));

  time = std::clock();
  w_cuimg<<<dimGrid, dimBlock>>>((i_float4*)buffer);
  cuimg::check_cuda_error();
  cudaMemcpy(buffer_cpu, buffer, sizeof(float4), cudaMemcpyDeviceToHost);
  cuimg::check_cuda_error();
  std::cout << "cuimg: "<< (float(clock() - time) / float(CLOCKS_PER_SEC)) << std::endl;

  std::cout << buffer_cpu[0].x << ", " << buffer_cpu[0].y << ", " << buffer_cpu[0].z << ", " << buffer_cpu[0].w << std::endl;
}
