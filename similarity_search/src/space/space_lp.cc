/**
 * Non-metric Space Library
 *
 * Authors: Bilegsaikhan Naidan (https://github.com/bileg), Leonid Boytsov (http://boytsov.info).
 * With contributions from Lawrence Cayton (http://lcayton.com/) and others.
 *
 * For the complete list of contributors and further details see:
 * https://github.com/searchivarius/NonMetricSpaceLib 
 * 
 * Copyright (c) 2014
 *
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#include <cmath>
#include <fstream>
#include <string>
#include <sstream>

#include "space/space_lp.h"
#include "logging.h"
#include "experimentconf.h"

namespace similarity {

template <typename dist_t>
dist_t SpaceLp<dist_t>::HiddenDistance(const Object* obj1, const Object* obj2) const {
  CHECK(obj1->datalength() > 0);
  CHECK(obj1->datalength() == obj2->datalength());
  const dist_t* x = reinterpret_cast<const dist_t*>(obj1->data());
  const dist_t* y = reinterpret_cast<const dist_t*>(obj2->data());
  const size_t length = obj1->datalength() / sizeof(dist_t);

  return distObj_(x, y, length);
}

template <typename dist_t>
std::string SpaceLp<dist_t>::ToString() const {
  std::stringstream stream;
  stream << "SpaceLp: p = " << distObj_.getP() << " (custom implement.) = " << distObj_.getCustom();
  return stream.str();
}

template class SpaceLp<float>;
template class SpaceLp<double>;

}  // namespace similarity
