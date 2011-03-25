#include <cassert>
#include <cuimg/point2d.h>

using namespace cuimg;

int main()
{
  point2d<int> p(300, 200);

  assert(p.row() == 300);
  assert(p.col() == 200);

  point2d<int> o(3, 2);

  p = o;
  assert(p.row() == 3);
  assert(p.col() == 2);

}
