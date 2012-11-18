/*=========================================================================

  Program: Bender

  Copyright (c) Kitware Inc.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0.txt

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

=========================================================================*/

#ifndef __WeightMap_h
#define __WeightMap_h

// .NAME WeightMap - sparse representation over weight vectors over a volume
// .SECTION General Description
// This is a sparse representation for an image of vector-weights. Only
// non-zero entries are represented.

#include <itkImage.h>
#include <itkVariableLengthVector.h>
#include <vector>
#include <limits>
#include "BenderCommonExport.h"

namespace bender
{
class BENDER_COMMON_EXPORT WeightMap
{
 public:
  typedef unsigned char SiteIndex;
  struct WeightEntry
  {
    SiteIndex Index;
    float Value;

  WeightEntry(): Index(std::numeric_limits<std::size_t>::max()), Value(0){}
  };

  typedef itk::Index<3> Voxel;
  typedef std::vector<SiteIndex> RowSizes;
  typedef std::vector<WeightEntry> WeightEntries;
  typedef itk::ImageRegion<3> Region;
  typedef itk::VariableLengthVector<float> WeightVector;

  typedef std::vector<WeightEntries> WeightLUT; //for any j, WeightTable[...][j] correspond to
  //the weights at a vixel

  typedef itk::Image<size_t,3> WeightLUTIndex; //for each voxel v, WeightLUTIndex[v] index into the
  //the "column" of WeightLUT


  WeightMap();
  void Init(const std::vector<Voxel>& voxels, const Region& region);
  bool Insert(const Voxel& v, SiteIndex index, float value);
  void Get(const Voxel& v, WeightVector& values) const;
  void AddRow();
  void Print() const;

 private:
  WeightLUT LUT;
  WeightLUTIndex::Pointer LUTIndex;
  RowSizes RowSize;
  size_t Cols;
};
};

#endif
