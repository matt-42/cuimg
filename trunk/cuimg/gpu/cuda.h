#ifndef CUIMG_CUDA_H_
# define CUIMG_CUDA_H_

# ifndef NO_CUDA
#  include <GL/gl.h>
#  include <cuda.h>
#  include <cuda_runtime.h>
#  include <host_defines.h>
#  include <cudaGL.h>
#  include <cuda_gl_interop.h>
# else
#  include <cuimg/gpu/nocuda_defines.h>
# endif

#endif // ! CUIMG_NOCUDA_DEFINES_H_
