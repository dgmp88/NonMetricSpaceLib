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

#include "space.h"
#include "rangequery.h"
#include "knnquery.h"
#include "method/dummy.h"

namespace similarity {

template <typename dist_t>
void DummyMethod<dist_t>::Search(RangeQuery<dist_t>* query) {
  if (bDoSeqSearch_) {
    for (size_t i = 0; i < data_.size(); ++i) {
      query->CheckAndAddToResult(data_[i]);
    }
  } else {
    for (int i =0; i < 100000; ++i);
  }
}

template <typename dist_t>
void DummyMethod<dist_t>::Search(KNNQuery<dist_t>* query) {
  if (bDoSeqSearch_) {
    for (size_t i = 0; i < data_.size(); ++i) {
      query->CheckAndAddToResult(data_[i]);
    }
  } else {
    for (int i =0; i < 100000; ++i);
  }
}

template <typename dist_t>
void 
DummyMethod<dist_t>::SetQueryTimeParamsInternal(AnyParamManager& pmgr) {
  int dummy;
  pmgr.GetParamOptional("dummyParam", dummy);
  LOG(LIB_INFO) << "Set dummy = " << dummy;
  pmgr.CheckUnused();
}

template <typename dist_t>
vector<string>
DummyMethod<dist_t>::GetQueryTimeParamNames() const {
  vector<string> names;
  names.push_back("dummyParam");
  return names;
}

template class DummyMethod<float>;
template class DummyMethod<double>;
template class DummyMethod<int>;

}
