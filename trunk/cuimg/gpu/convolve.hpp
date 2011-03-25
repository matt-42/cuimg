#ifndef CUIMG_CONVOLVE_HPP_
# define CUIMG_CONVOLVE_HPP_

# include <cuimg/gpu/convolve.h>
# include <cuimg/util.h>
# include <cuimg/gpu/texture.h>

namespace cuimg
{
  namespace conv_internals
  {
    template <typename T>
    struct conv_input_tex;
      REGISTER_TEXTURE2D_PROXY(conv_input_tex);

    texture<float, 1, cudaReadModeElementType> tex_weights;
    texture<int2, 1, cudaReadModeElementType> tex_dpoints;

    template <typename I, typename O>
    __global__ void convolve_kernel(I, kernel_image2d<O> out, unsigned kernelsize)
    {
      i_int2 p = thread_pos2d();

      if (!out.has(p))
        return;

      bt_change_vtype(O, type_mult(bt_vtype(O), float)) r  = zero();
      for(int i = 0; i < kernelsize; i++)
      {
        float w = tex1Dfetch(tex_weights, i);
        point2d<int> n = i_int2(tex1Dfetch(tex_dpoints, i)) + p;
        if (out.has(n))
          r += O(tex2D(conv_input_tex<I>::tex(), n)) * w;
      }
      out(p) = r;
    }

  }

  template <typename U, typename V, typename WW>
  void convolve(const image2d<U>& in, image2d<V>& out,
                const WW& weighted_window, dim3 dimblock)
  {
    using namespace conv_internals;

    unsigned size = weighted_window.size();
    bindTexture2d(in, conv_internals::conv_input_tex<U>::tex());
    float* weights = new float[size];
    i_int2* dpoints = new i_int2[size];
    for (unsigned i = 0; i < size; i++)
    {
      weights[i] = weighted_window.weights(i);
      dpoints[i] = weighted_window.dpoints(i);
    }

    float* weights_cuda;
    i_int2* dpoints_cuda;
    cudaMalloc(&weights_cuda, size * sizeof(float));
    cudaMalloc(&dpoints_cuda, size * sizeof(i_int2));
    check_cuda_error();

    cudaMemcpy(weights_cuda, weights, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dpoints_cuda, dpoints, size * sizeof(i_int2), cudaMemcpyHostToDevice);
    check_cuda_error();

    delete[] weights;
    delete[] dpoints;

    cudaBindTexture(0, tex_weights, weights_cuda,
                    cudaCreateChannelDesc<float>(), size * sizeof(float));
    cudaBindTexture(0, tex_dpoints, dpoints_cuda,
                    cudaCreateChannelDesc<int2>(), size * sizeof(i_int2));
    check_cuda_error();
    dim3 dimgrid = grid_dimension(in.domain(), dimblock);
    convolve_kernel<<<dimgrid, dimblock>>>(U(), mki(out), size);
    cudaFree(weights_cuda);
    cudaFree(dpoints_cuda);
  }

//   template <typename U, typename V, typename WW>
//   void convolveRows(const image2d<U>& in, const image2d<V>& out, const WW& weighted_window)
//   {
//     using namespace conv_internals;
//   }

//   template <typename U, typename V, typename WW>
//   void convolveCols(const image2d<U>& in, const image2d<V>& out, const WW& weighted_window)
//   {
//     using namespace conv_internals;
//   }

}

#endif
