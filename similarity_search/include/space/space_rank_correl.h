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

#ifndef _RANK_CORREL_SPACE_H_
#define _RANK_CORREL_SPACE_H_

#include <string>
#include <map>
#include <stdexcept>

#include <string.h>
#include "global.h"
#include "object.h"
#include "utils.h"
#include "space.h"
#include "space_vector.h"
#include "permutation_type.h"

namespace similarity {

template <PivotIdType (*RankCorrelDistFunc)(const PivotIdType*, const PivotIdType*, size_t)>
class RankCorrelVectorSpace : public VectorSpace<PivotIdType> {
 public:

  virtual std::string ToString() const { return "rank correlation vector space"; }
 protected:
  virtual Space<PivotIdType>* HiddenClone() const { return new RankCorrelVectorSpace<RankCorrelDistFunc>(); } // no parameters 
  // Should not be directly accessible
  virtual PivotIdType HiddenDistance(const Object* obj1, const Object* obj2) const {
    CHECK(obj1->datalength() > 0);
    CHECK(obj1->datalength() == obj2->datalength());
    const PivotIdType* x = reinterpret_cast<const PivotIdType*>(obj1->data());
    const PivotIdType* y = reinterpret_cast<const PivotIdType*>(obj2->data());
    const size_t length = obj1->datalength() / sizeof(size_t);

    return RankCorrelDistFunc(x, y, length);
  }
};

}  // namespace similarity

#endif
