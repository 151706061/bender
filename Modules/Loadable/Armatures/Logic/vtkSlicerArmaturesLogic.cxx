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

// Models includes
#include "vtkSlicerModelsLogic.h"

// Armatures includes
#include "vtkSlicerArmaturesLogic.h"

// MRML includes

// VTK includes
#include <vtkMath.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkXMLDataElement.h>
#include <vtkXMLDataParser.h>

// STD includes
#include <cassert>

vtkCxxSetObjectMacro(vtkSlicerArmaturesLogic, ModelsLogic, vtkSlicerModelsLogic);

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkSlicerArmaturesLogic);

//----------------------------------------------------------------------------
vtkSlicerArmaturesLogic::vtkSlicerArmaturesLogic()
{
  this->ModelsLogic = 0;
}

//----------------------------------------------------------------------------
vtkSlicerArmaturesLogic::~vtkSlicerArmaturesLogic()
{
}

//----------------------------------------------------------------------------
void vtkSlicerArmaturesLogic::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//----------------------------------------------------------------------------
vtkMRMLModelNode* vtkSlicerArmaturesLogic::AddArmatureFile(const char* fileName)
{
  vtkNew<vtkXMLDataParser> armatureParser;
  armatureParser->SetFileName(fileName);
  int res = armatureParser->Parse();
  if (res == 0)
    {
    vtkErrorMacro(<<"Fail to read" << fileName << ". Not a valid armature file");
    return 0;
    }
  vtkNew<vtkPolyData> armature;
  vtkNew<vtkPoints> points;
  points->SetDataTypeToDouble();
  armature->SetPoints(points.GetPointer());
  armature->Allocate(100);
  vtkXMLDataElement* armatureElement = armatureParser->GetRootElement();
  double orientationXYZW[4];
  armatureElement->GetVectorAttribute("orientation", 4, orientationXYZW);
  double orientationWXYZ[4] = {orientationXYZW[3], orientationXYZW[0],
                               orientationXYZW[1], orientationXYZW[2]};

  double origin[3] = {0., 0., 0.};
  for (int child = 0; child < armatureElement->GetNumberOfNestedElements(); ++child)
    {
    this->ReadBone(armatureElement->GetNestedElement(child), armature.GetPointer(), origin, orientationWXYZ);
    }
  return this->ModelsLogic->AddModel(armature.GetPointer());
}

//----------------------------------------------------------------------------
void vtkSlicerArmaturesLogic::ReadBone(vtkXMLDataElement* boneElement,
                                       vtkPolyData* polyData,
                                       const double origin[3],
                                       const double parentOrientation[4])
{
  double parentMatrix[3][3];
  vtkMath::QuaternionToMatrix3x3(parentOrientation, parentMatrix);

  double localHead[3] = {0., 0., 0.};
  boneElement->GetVectorAttribute("head", 3, localHead);
  double head[3] = {0., 0., 0.};
  vtkMath::Multiply3x3(parentMatrix, localHead, head);

  double parentPosedOrientation[4] =
    {parentOrientation[0], parentOrientation[1],
     parentOrientation[2], parentOrientation[3]};
  bool applyPose = true;
  if (applyPose)
    {
    vtkXMLDataElement * poseElement =
      boneElement->FindNestedElementWithName("pose");
    if (poseElement)
      {
      this->ReadPose(poseElement, parentPosedOrientation, true);
      }
    }

#ifndef BEFORE
  double parentPosedMatrix[3][3];
  vtkMath::QuaternionToMatrix3x3(parentPosedOrientation, parentPosedMatrix);

  double localTail[3] = {0., 0. ,0};
  boneElement->GetVectorAttribute("tail", 3, localTail);
  double tail[3] = {0., 0. ,0};
  vtkMath::Subtract(localTail, localHead, tail);
  vtkMath::Multiply3x3(parentPosedMatrix, tail, tail);
  vtkMath::Add(localHead, tail, tail);

  vtkMath::Add(origin, head, head);
  vtkMath::Add(origin, tail, tail);

  vtkIdType indexes[2];
  indexes[0] = polyData->GetPoints()->InsertNextPoint(head);
  indexes[1] = polyData->GetPoints()->InsertNextPoint(tail);

  polyData->InsertNextCell(VTK_LINE, 2, indexes);
#endif

  double localOrientationXYZW[4] = {0., 0., 0., 1.};
  boneElement->GetVectorAttribute("orientation", 4, localOrientationXYZW);
  double localOrientationWXYZ[4] = {localOrientationXYZW[3], localOrientationXYZW[0],
                                    localOrientationXYZW[1], localOrientationXYZW[2]};
  double worldOrientation[4];
  //vtkMath::MultiplyQuaternion(parentOrientation, localOrientationWXYZ, worldOrientation);
  vtkMath::MultiplyQuaternion(parentPosedOrientation, localOrientationWXYZ, worldOrientation);
  //vtkMath::MultiplyQuaternion(localOrientationWXYZ, parentOrientation, worldOrientation);
  //vtkMath::MultiplyQuaternion(localOrientationWXYZ, parentPosedOrientation, worldOrientation);
/*
  vtkXMLDataElement * poseElement =
    boneElement->FindNestedElementWithName("pose");
  if (applyPose)
    {
    if (poseElement)
      {
      this->ReadPose(poseElement, worldOrientation, false);
      }
    }
*/
#ifdef BEFORE
  double parentPosedMatrix[3][3];
  vtkMath::QuaternionToMatrix3x3(worldOrientation, parentPosedMatrix);

  double localTail[3] = {0., 0. ,0};
  boneElement->GetVectorAttribute("tail", 3, localTail);
  double tail[3] = {0., 0. ,0};
  vtkMath::Subtract(localTail, localHead, tail);
  vtkMath::Multiply3x3(parentPosedMatrix, tail, tail);
  vtkMath::Add(localHead, tail, tail);

  vtkMath::Add(origin, head, head);
  vtkMath::Add(origin, tail, tail);

  vtkIdType indexes[2];
  indexes[0] = polyData->GetPoints()->InsertNextPoint(head);
  indexes[1] = polyData->GetPoints()->InsertNextPoint(tail);

  polyData->InsertNextCell(VTK_LINE, 2, indexes);
#endif

  std::cout << "local quat: w=" << localOrientationWXYZ[0] << " x=" << localOrientationWXYZ[1] << " y=" << localOrientationWXYZ[2] << " z=" << localOrientationWXYZ[3] << std::endl;
  std::cout << "New quat: w=" << worldOrientation[0] << " x=" << worldOrientation[1] << " y=" << worldOrientation[2] << " z=" << worldOrientation[3] << std::endl;
  for (int child = 0; child < boneElement->GetNumberOfNestedElements(); ++child)
    {
    vtkXMLDataElement* childElement = boneElement->GetNestedElement(child);
    if (strcmp(childElement->GetName(),"bone") == 0)
      {
      this->ReadBone(childElement, polyData, tail, worldOrientation);
      }
    else if (strcmp(childElement->GetName(),"pose") == 0)
      {
      // already parsed
      }
    else
      {
      vtkWarningMacro( << "XML element " << childElement->GetName()
                       << "is not supported");
      }
    }
}

//----------------------------------------------------------------------------
void vtkSlicerArmaturesLogic::ReadPose(vtkXMLDataElement* poseElement,
                                       double parentOrientation[4], bool invert)
{
  double poseRotationXYZW[4] = {0., 0., 0., 0.};
  poseElement->GetVectorAttribute("rotation", 4, poseRotationXYZW);
  double poseRotationWXYZ[4] = {poseRotationXYZW[3], poseRotationXYZW[0],
                                poseRotationXYZW[1], poseRotationXYZW[2]};
  if (invert)
    {
    vtkMath::MultiplyQuaternion(poseRotationWXYZ, parentOrientation, parentOrientation);
    }
  else
    {
    vtkMath::MultiplyQuaternion(parentOrientation, poseRotationWXYZ, parentOrientation);
    }
}
