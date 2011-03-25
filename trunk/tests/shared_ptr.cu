
#include <boost/shared_ptr.hpp>

int main()
{
  boost::shared_ptr<int> a;

  int* p = a.get();
}