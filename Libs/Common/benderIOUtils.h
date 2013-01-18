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

#ifndef __benderIOUtils_h
#define __benderIOUtils_h

// .NAME BenderIOUtils - convenience functions read and write to files
// .SECTION General Description
//  A set of utility functions to read and write files

// Bender includes
#include "BenderCommonExport.h"
#include <string>

class vtkPolyData;
namespace bender
{
class BENDER_COMMON_EXPORT IOUtils
{
 public:
  /// Read .vtk/.stl/.vtp file into a vtkPolyData.
  /// Negate x and y coordinates if invertXY==true
  /// The caller of the function is responsible for deleting the returned polydata.
  static vtkPolyData* ReadPolyData(const std::string& fileName, bool invertXY=false);

  static void WritePolyData(vtkPolyData* polyData, const std::string& fileName);

  /// Convenient method to write an itk image to disk.
  template <class ImageType>
  static void WriteImage(typename ImageType::Pointer image,const char* fname);

  static void FilterStart(const char* filterName, const char* comment =0);
  static void FilterProgress(const char* filterName, float progress,
                             double fraction = 1.0, double start = 0.);
  static void FilterEnd(const char* filterName, size_t time = 0);
};

};
#include "benderIOUtils.txx"

#endif
