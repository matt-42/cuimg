#ifndef CUIMG_COMPACT_PARTICLES_IMAGE_H_
# define CUIMG_COMPACT_PARTICLES_IMAGE_H_

# include <vector>
# include <omp.h>
# include <cuimg/cpu/host_image2d.h>

namespace cuimg
{

  template <typename P, typename PP>
  inline void compact_particles_image(const host_image2d<P>& image,
				      std::vector<PP>& v)
  {
    SCOPE_PROF(compact_particles);
    v.clear();


    unsigned size = 0;

    int num_threads = omp_get_max_threads();

    static std::vector<std::vector<PP> > compaction_openmp_buffers(num_threads, std::vector<PP>(20000));
    std::vector<int> out_size(num_threads);

    START_PROF(loop);
#pragma omp parallel
    {
      int tid;
      PP* curr;
      tid = omp_get_thread_num();
      curr = &(compaction_openmp_buffers[tid][0]);
      // std::cout << "curr start: " << curr << std::endl;
      assert(curr);
#pragma omp for schedule(static, 4)
      for (unsigned r = 0; r < image.nrows(); r++)
      {
        const P* it = &image(r, 0);
        const P* end = it + image.ncols();
        for (unsigned c = 0; it < end; it++, c++)
          if (it->age != 0)
          {
            assert(curr >= compaction_openmp_buffers[tid] && curr < compaction_openmp_buffers[tid] + 20000);
            //*curr = i_short2(r, c);
            curr->set(*it, i_short2(r, c));
            // std::cout << "curr write: " << curr <<  " " << tid << std::endl;

            ++curr;
          }
      }

      out_size[tid] = curr - &(compaction_openmp_buffers[tid][0]);

#pragma omp barrier
      unsigned before = 0;
      for (unsigned t = 0; t < tid; t++)
        before += out_size[t];

      if (tid == 0)
      {
        unsigned size = 0;
        for (unsigned t = 0; t < num_threads; t++)
          size += out_size[t];
        v.resize(size);
      }

#pragma omp barrier
      memcpy(&v[0] + before, &(compaction_openmp_buffers[tid][0]), out_size[tid] * sizeof(int));
    }
    END_PROF(loop);
  }

}

#endif
