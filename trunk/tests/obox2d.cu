
#include <cassert>
#include <cuimg/point2d.h>
#include <cuimg/obox2d.h>

using namespace cuimg;

int main()
{
  obox2d<point2d<int> > b(300, 200);

  assert(b.nrows() == 300);
  assert(b.ncols() == 200);

  assert(b.has(point2d<int>(0, 0)));
  assert(!b.has(point2d<int>(-1, 0)));
  assert(!b.has(point2d<int>(0, -1)));
  assert(b.has(point2d<int>(299, 199)));
  assert(!b.has(point2d<int>(300, 199)));
  assert(!b.has(point2d<int>(299, 200)));
}
