#ifndef CUIMG_COMPACT_PARTICLES_IMAGE_H_
# define CUIMG_COMPACT_PARTICLES_IMAGE_H_

# include <vector>
# include <omp.h>
# include <cuimg/cpu/host_image2d.h>
# include <cuimg/profiler.h>

namespace cuimg
{

  template <typename P, typename PP>
  inline void compact_particles_image(host_image2d<P>& image,
				      std::vector<PP>& v)
  {
    SCOPE_PROF(compact_particles);
    v.clear();


    unsigned size = 0;

    int num_threads = omp_get_max_threads();

    static std::vector<std::vector<PP> > compaction_openmp_buffers(num_threads, std::vector<PP>(100000));
    std::vector<int> out_size(num_threads);

    START_PROF(loop);
#pragma omp parallel
    {
      int tid;
      PP* curr;
      tid = omp_get_thread_num();
      std::vector<PP>& buffer = compaction_openmp_buffers[tid];
      curr = &(buffer.front());
      PP*buf_end = &(buffer.back());
      assert(curr);
#pragma omp for schedule(static, image.nrows() / num_threads)
      for (short r = 0; r < image.nrows(); r++)
      {
        const P* it = &image(r, 0);
        const P* row_end = it + image.ncols();
        for (short c = 0; it < row_end; it++, c++)
          if (it->age != 0 && curr < buf_end)
          {
            assert(curr < buf_end);
            curr->set(*it, i_short2(r, c));
	    assert(image.has(curr->pos));
            ++curr;
          }
      }

      out_size[tid] = curr - &(buffer.front());

#pragma omp barrier
      unsigned before = 0;
      for (unsigned t = 0; t < tid; t++)
        before += out_size[t];

      if (tid == 0)
      {
        unsigned size = 0;
        for (unsigned t = 0; t < num_threads; t++)
        {
          size += out_size[t];
        }
        v.resize(size);
      }

#pragma omp barrier
      memcpy(&v[0] + before, &(buffer[0]), out_size[tid] * sizeof(PP));

    }

#pragma omp parallel for schedule(static, v.size() / num_threads)
      for (unsigned i = 0; i < v.size(); i++)
	image(v[i].pos).vpos = i;

    END_PROF(loop);
  }

}

#endif
