/**
 * Non-metric Space Library
 *
 * Authors: Bilegsaikhan Naidan (https://github.com/bileg), Leonid Boytsov (http://boytsov.info).
 * With contributions from Lawrence Cayton (http://lcayton.com/) and others.
 *
 * For the complete list of contributors and further details see:
 * https://github.com/searchivarius/NonMetricSpaceLib 
 * 
 * This is a wrapper class for the Wei Dong implementation of https://code.google.com/p/nndes/,
 * which also contains some of the original code from Wei Dong's repository.  
 * Wei Dong, Charikar Moses, and Kai Li. 2011. Efficient k-nearest neighbor graph construction for generic similarity measures. 
 * In Proceedings of the 20th international conference on World wide web (WWW '11). ACM, New York, NY, USA, 577-586. 
 * 
 * The Wei Dong's code can be redistributed given that the license (see below) is retained in the source code.
 */
/*
Copyright (C) 2010,2011 Wei Dong <wdong@wdong.org>
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#ifndef NNDES_METHOD_H_
#define NNDES_METHOD_H_

#include <string>
#include <sstream>
#include <memory>

#include "index.h"
#include "space.h"

#include "nndes/nndes-common.h"
#include "nndes/nndes.h"

#define METH_NNDES             "nndes"

namespace similarity {

using std::string;

template <typename dist_t>
class NNDescentMethod : public Index<dist_t> {
 public:
  NNDescentMethod(bool PrintProgress,
                  const Space<dist_t>* space, 
                  const ObjectVector& data, 
                  const AnyParams& AllParams);
  ~NNDescentMethod(){};

  /* 
   * Just the name of the method, consider printing crucial parameter values
   */
  const std::string ToString() const { 
    stringstream str;
    str << "NNDescentMethod method: ";
    return str.str();
  }

  void Search(RangeQuery<dist_t>* query);
  void Search(KNNQuery<dist_t>* query);

  virtual vector<string> GetQueryTimeParamNames() const;

  class SpaceOracle {
  public:
    SpaceOracle(const Space<dist_t>* space, const ObjectVector& data) :
                space_(space), data_(data) {}
    inline dist_t operator()(IdType id1, IdType id2) const {
      return space_->IndexTimeDistance(data_.at(id1), data_.at(id2));
    }
  private:
    const Space<dist_t>*    space_;
    const ObjectVector&     data_;
  };
 private:
  void SearchGreedy(KNNQuery<dist_t>* query);
  void SearchSmallWorld(KNNQuery<dist_t>* query);

  typedef pair<dist_t, IdType>      EvaluatedNode;

  virtual void SetQueryTimeParamsInternal(AnyParamManager& );

  const Space<dist_t>*    space_;
  const ObjectVector&     data_;
  size_t                  NN_; // K in the original Wei Dong's code nndes.cpp
  size_t                  searchNN_;
  size_t                  controlQty_; // control in the original Wei Dong's code nndes.cpp
  size_t                  iterationQty_; // iteration in the original Wei Dong's code nndes.cpp
  float                   rho_;
  float                   delta_;

  SpaceOracle                        nndesOracle_;
  unique_ptr<NNDescent<SpaceOracle>> nndesObj_;

  size_t                  initSearchAttempts_;
  bool                    greedy_;
  // disable copy and assign
  DISABLE_COPY_AND_ASSIGN(NNDescentMethod);
};

}   // namespace similarity

#endif
