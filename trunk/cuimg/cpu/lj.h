#ifndef CUIMG_CPU_LJ_H_
# define CUIMG_CPU_LJ_H_

# include <vector>
//# include <cuimg/cpu/host_image2d.h>

namespace cuimg
{
  template <unsigned S, unsigned N>
  unsigned lj_mapping(unsigned s, unsigned i, unsigned j)
  {
    return lj<float, S, N>::scale_size * s + ((N - i) * ((N - i) + 1) / 2) + j;
  }

  template <unsigned S, unsigned N>
  unsigned lj_size()
  {
    return lj<float, S, N>::size;
  }

  // S: number of scales.
  // N: max derivative.
  template <typename T, unsigned S, unsigned N>
  struct lj
  {
    typedef lj<T, S, N> self;
    enum { scale_size = ((N + 1) * (N + 2) / 2),
           size       = scale_size * S};

    lj()
    {
    }

    lj(const self& e)
    {
      *this = e;
    }

    self& operator=(const self& e)
    {
      memcpy((char*)data, (char*)e.data, size * sizeof(T));
      return *this;
    }

    self& operator=(const zero&)
    {
      for (unsigned i = 0; i < size; i++)
        data[i] = zero();
      return *this;
    }

    const T& operator()(unsigned s, unsigned i, unsigned j) const
    {
      return data[lj_mapping<S, N>(s, i, j)];
    }

    T& operator()(unsigned s, unsigned i, unsigned j)
    {
      return data[lj_mapping<S, N>(s, i, j)];
    }

    T& operator[](unsigned idx)
    {
      return data[idx];
    }
    const T& operator[](unsigned idx) const
    {
      return data[idx];
    }

    T data[size];
  };

  template <typename T, unsigned S, unsigned N>
  void lj_fill(const host_image2d<T>& in, host_image2d<lj<T, S, N> >& out)
  {
    typedef lj<T, S, N> F;

    assert(in.domain() == out.domain());
    for (unsigned r = 0; r < in.nrows(); r++)
    for (unsigned c = 0; c < in.ncols(); c++)
    {
      for (unsigned i = 0; i < F::size; i++)
        out(r, c)[i] = in(r, c);
    }
  }

  template <typename T, typename U, unsigned S, unsigned N, typename WW>
  void lj_convolve_cols(const host_image2d<T>& in, host_image2d<lj<U, S, N> >& out,
                        const std::vector<WW>& wws)
  {
    typedef lj<U, S, N> F;

    assert(in.domain() == out.domain());

    int hs = wws[0].size() / 2;

//#pragma omp parallel for
    for (int r = 0; r < int(in.nrows()); r++)
    for (unsigned c = 0; c < in.ncols(); c++)
    {
      out(r, c) = zero();
      for (int i = r - hs; i < int(r + hs); i++)
      {
        if (i >= 0 && i < int(in.nrows()))
        {
          for (unsigned w = 0; w < F::size; w++)
            out(r, c)[w] += in(i, c)[w] * wws[w][i - (r - hs)];
        }
      }
    }
  }

  template <typename T, typename U, unsigned S, unsigned N, typename WW>
  void lj_convolve_rows(const host_image2d<T>& in, host_image2d<lj<U, S, N> >& out,
                        const std::vector<WW>& wws)
  {
    typedef lj<U, S, N> F;

    assert(in.domain() == out.domain());

    int hs = wws[0].size() / 2;
//#pragma omp parallel for
    for (int r = 0; r < int(in.nrows()); r++)
    for (unsigned c = 0; c < in.ncols(); c++)
    {
      out(r, c) = zero();
      for (int i = c - hs; i < int(c + hs); i++)
      {
        if (i >= 0 && i < int(in.ncols()))
        {
          for (unsigned w = 0; w < F::size; w++)
            out(r, c)[w] += in(r, i)[w] * wws[w][i - (c - hs)];
        }
      }
    }
  }

  template <typename T, typename U, unsigned S, unsigned N>
  void lj_extract_comp(const host_image2d<lj<T, S, N> >& in,
                       host_image2d<U>& out,
                       unsigned i)
  {
    for (unsigned r = 0; r < in.nrows(); r++)
    for (unsigned c = 0; c < in.ncols(); c++)
      out(r, c) = in(r, c)[i];
  }

  template <typename WW, unsigned S, unsigned N>
  void lj_make_kernels(std::vector<WW>& ww_rows, std::vector<WW>& ww_cols,
                       const std::vector<int>& scales)
  {
    unsigned size = lj_size<S, N>();
    ww_rows.resize(size);
    ww_cols.resize(size);

    unsigned wsize = scales[S-1] * 6 + 1;

    for (unsigned s = 0; s < S; s++)
    for (unsigned i = 0; i <= N; i++)
    for (unsigned j = 0; i+j <= N; j++)
    {
      make_gaussian_kernel_1d(i, float(scales[s]), ww_rows[lj_mapping<S, N>(s, i, j)], wsize);
      make_gaussian_kernel_1d(j, float(scales[s]), ww_cols[lj_mapping<S, N>(s, i, j)], wsize);
    }

  }

  template <typename T, unsigned S, unsigned N>
  void lj_normalize(host_image2d<lj<T, S, N> >& ima, const std::vector<int>& scales)
  {
    typedef lj<T, S, N> F;
    float coefs[F::size];

    for (unsigned s = 0; s < S; s++)
      for (unsigned i = 0; i <= N; i++)
        for (unsigned j = 0; i+j <= N; j++)
          //coefs[lj_mapping<S, N>(s, i, j)] = std::powf(float(scales[s] + (i + j)), float((i + j))) / float(i + j + 1);
          coefs[lj_mapping<S, N>(s, i, j)] = std::powf(float(scales[s]), float((i + j))) / float(i + j + 1);

    for (unsigned r = 0; r < ima.nrows(); r++)
    for (unsigned c = 0; c < ima.ncols(); c++)
    {
      for (unsigned i = 0; i < F::size; i++)
        ima(r, c)[i] = ima(r, c)[i] * coefs[i];
    }
  }

  template <typename T, unsigned S, unsigned N>
  float lj_single_scale_distance(const lj<T, S, N>& a, const lj<T, S, N>& b, const int s)
  {
    T d = zero();
    for (unsigned i = 0; i <= N; i++)
      for (unsigned j = 0; i+j <= N; j++)
      {
        T x = a(s, i, j) - b(s, i, j);
        for (unsigned k = 0; k < T::size; k++)
          d[k] = d[k] + x[k] * x[k];
      }

    //d = sqrt(d);

    float res = 0.f;
    for (unsigned k = 0; k < T::size; k++)
      res += d[k];
    return res / float(T::size * lj<T, S, N>::scale_size);
  }

  template <typename T, unsigned S, unsigned N>
  float lj_pan_scalic_distance(const lj<T, S, N>& a, const lj<T, S, N>& b)
  {
    T d = zero();
    for (unsigned s = 0; s < S; s++)
      for (unsigned i = 0; i <= N; i++)
        for (unsigned j = 0; i+j <= N; j++)
        {
          T x = a(s, i, j) - b(s, i, j);
          for (unsigned k = 0; k < T::size; k++)
            d[k] = d[k] + x[k] * x[k];
        }

//    d = sqrt(d);


    float res = 0.f;
    for (unsigned k = 0; k < T::size; k++)
      res += d[k];
    return res / float(T::size * lj<T, S, N>::size);
  }

}

#endif
