/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkPolyDataPointSampler.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
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

#include "vtkPolyDataPointSampler.h"

#include "vtkCellArray.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkPolyData.h"
#include "vtkEdgeTable.h"
#include "vtkMath.h"

vtkStandardNewMacro(vtkPolyDataPointSampler);

//------------------------------------------------------------------------
vtkPolyDataPointSampler::vtkPolyDataPointSampler()
{
  this->Distance = 0.01;
  this->Distance2 = this->Distance * this->Distance;

  this->GenerateVertexPoints = 1;
  this->GenerateEdgePoints = 1;
  this->GenerateInteriorPoints = 1;
  
  this->GenerateVertices = 1;

  this->CurrentCellId = -1;
}

//------------------------------------------------------------------------
int vtkPolyDataPointSampler::RequestData(vtkInformation *vtkNotUsed(request),
                                         vtkInformationVector **inputVector,
                                         vtkInformationVector *outputVector)
{
  // get the info objects
  vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
  vtkInformation *outInfo = outputVector->GetInformationObject(0);

  // get the input and output
  vtkPolyData *input = vtkPolyData::SafeDownCast(
    inInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkPolyData *output = vtkPolyData::SafeDownCast(
    outInfo->Get(vtkDataObject::DATA_OBJECT()));

  // Initialize
  vtkDebugMacro(<<"Resampling polygonal data");

  if ( this->Distance <= 0.0 )
    {
    vtkWarningMacro("Cannot resample to zero distance\n");
    return 1;
    }
  if ( !input || !input->GetPoints() ||
       (!this->GenerateVertexPoints && !this->GenerateEdgePoints && !this->GenerateInteriorPoints) )
    {
    return 1;
    }
  vtkPoints *inPts = input->GetPoints();
  vtkIdType numInputPts = input->GetNumberOfPoints();
  
  // Prepare output
  int abort;
  double x0[3], x1[3];
  vtkIdType i, *pts, npts;
  this->InitializeOutput(output, input);
  this->Distance2 = this->Distance*this->Distance;

  // First the vertex points
  if ( this->GenerateVertexPoints )
    {
    for (vtkIdType i = 0; i < input->GetPoints()->GetNumberOfPoints(); ++i)
      {
      this->InsertNextPoint(output, input->GetPoint(i), input->GetPoints(), 1, &i, 0, 0);
      }
    }
  this->UpdateProgress (0.1);
  abort = this->GetAbortExecute();
  
  // Now the edge points
  vtkCellArray *inPolys = input->GetPolys();
  vtkCellArray *inStrips = input->GetStrips();
  if ( this->GenerateEdgePoints && !abort )
    {
    vtkEdgeTable *eTable = vtkEdgeTable::New();
    eTable->InitEdgeInsertion(numInputPts);
    vtkCellArray *inLines = input->GetLines();
    
    vtkIdType edgePoints[2];
    vtkIdType& p0 = edgePoints[0];
    vtkIdType& p1 = edgePoints[1];
    for ( inLines->InitTraversal(); inLines->GetNextCell(npts,pts); )
      {
      for (i=0; i<(npts-1); i++)
        {
        p0 = pts[i];
        p1 = pts[i+1];
        if ( eTable->IsEdge(p0, p1) == -1 )
          {
          eTable->InsertEdge(p0, p1);
          this->SampleEdge(output,inPts, edgePoints);
          }
        }
      }

    this->UpdateProgress (0.2);
    abort = this->GetAbortExecute();

    for ( inPolys->InitTraversal(); inPolys->GetNextCell(npts,pts); )
      {
      for (i=0; i<npts; i++)
        {
        p0 = pts[i];
        p1 = pts[(i+1)%npts];
        if ( eTable->IsEdge(p0,p1) == -1 )
          {
          eTable->InsertEdge(p0,p1);
          this->SampleEdge(output, inPts, edgePoints);
          }
        }
      }

    this->UpdateProgress (0.3);
    abort = this->GetAbortExecute();

    for ( inStrips->InitTraversal(); inStrips->GetNextCell(npts,pts); )
      {
      // The first triangle
      for (i=0; i<3; i++)
        {
        p0 = pts[i];
        p1 = pts[(i+1)%3];
        if ( eTable->IsEdge(p0,p1) == -1 )
          {
          eTable->InsertEdge(p0,p1);
          this->SampleEdge(output, inPts, edgePoints);
          }
        }
      // Now the other triangles
      for (i=3; i<npts; i++)
        {
        p0 = pts[i-2];
        p1 = pts[i];
        if ( eTable->IsEdge(p0,p1) == -1 )
          {
          eTable->InsertEdge(p0,p1);
          this->SampleEdge(output, inPts, edgePoints);
          }
        p0 = pts[i-1];
        p1 = pts[i];
        if ( eTable->IsEdge(p0,p1) == -1 )
          {
          eTable->InsertEdge(p0,p1);
          this->SampleEdge(output,inPts, edgePoints);
          }
        }
      }
    eTable->Delete();
    }
  
  this->UpdateProgress (0.5);
  abort = this->GetAbortExecute();

  // Finally the interior points on polygons and triangle strips
  if ( this->GenerateInteriorPoints && !abort )
    {
    // First the polygons
    this->CurrentCellId = 0;
    for ( inPolys->InitTraversal(); inPolys->GetNextCell(npts,pts);
          ++this->CurrentCellId)
      {
      if ( npts == 3 )
        {
        this->SampleTriangle(output,inPts,pts);
        }
      else
        {
        this->SamplePolygon(output,inPts,npts,pts);
        }
      }

    this->UpdateProgress (0.75);
    abort = this->GetAbortExecute();


    this->CurrentCellId = 0;
    // Next the triangle strips
    for ( inStrips->InitTraversal(); 
          inStrips->GetNextCell(npts,pts) && !abort; )
      {
      vtkIdType stripPts[3];
      for (i=0; i<(npts-2); i++)
        {
        stripPts[0] = pts[i];
        stripPts[1] = pts[i+1];
        stripPts[2] = pts[i+2];
        this->SampleTriangle(output,inPts,stripPts);
        }
      }//for all strips
    this->CurrentCellId = -1;
    }//for interior points
  
  this->UpdateProgress (0.90);
  abort = this->GetAbortExecute();

  // Generate vertex cells if requested
  if ( this->GenerateVertices && !abort )
    {
    vtkIdType id, numPts = output->GetPoints()->GetNumberOfPoints();
    vtkCellArray *verts = vtkCellArray::New();
    verts->Allocate(numPts+1);
    verts->InsertNextCell(numPts);
    for ( id=0; id < numPts; id++)
      {
      verts->InsertCellPoint(id);
      }
    output->SetVerts(verts);
    verts->Delete();
    }

  return 1;
}

//------------------------------------------------------------------------
void vtkPolyDataPointSampler
::InitializeOutput(vtkPolyData *output, vtkPolyData* input)
{
  vtkPoints *newPts = input->GetPoints()->NewInstance();
  output->SetPoints(newPts);
  newPts->Delete();
}

//------------------------------------------------------------------------
void vtkPolyDataPointSampler
::SampleEdge(vtkPolyData *output, vtkPoints *inPts, vtkIdType *pts)
{
  double x0[3], x1[3];
  inPts->GetPoint(pts[0],x0);
  inPts->GetPoint(pts[1],x1);

  double len = vtkMath::Distance2BetweenPoints(x0,x1);
  if ( len > this->Distance2 )
    {
    double t, x[3];
    len = sqrt(len);
    int npts = static_cast<int>(len/this->Distance) + 2;
    for (vtkIdType id=1; id<(npts-1); id++)
      {
      t = static_cast<double>(id)/(npts-1);
      x[0] = x0[0] + t*(x1[0]-x0[0]);
      x[1] = x0[1] + t*(x1[1]-x0[1]);
      x[2] = x0[2] + t*(x1[2]-x0[2]);
      this->InsertNextPoint(output, x, inPts, 2, pts, t);
      }
    }
}
  
//------------------------------------------------------------------------
void vtkPolyDataPointSampler
::SampleTriangle(vtkPolyData *output, vtkPoints *inPts, vtkIdType *pts)
{
  double x0[3], x1[3], x2[3];
  inPts->GetPoint(pts[0],x0);
  inPts->GetPoint(pts[1],x1);
  inPts->GetPoint(pts[2],x2);

  double l1 = vtkMath::Distance2BetweenPoints(x0,x1);
  double l2 = vtkMath::Distance2BetweenPoints(x0,x2);
  if ( l1 > this->Distance2 || l2 > this->Distance2 )
    {
    double s, t, x[3];
    l1 = sqrt(l1);
    l2 = sqrt(l2);
    int n1 = static_cast<int>(l1/this->Distance) + 2;
    int n2 = static_cast<int>(l2/this->Distance) + 2;
    n1 = ( n1 < 3 ? 3 : n1); //make sure there is at least one point
    n2 = ( n2 < 3 ? 3 : n2);
    for (vtkIdType j=1; j<(n2-1); j++)
      {
      t = static_cast<double>(j)/(n2-1);
      for (vtkIdType i=1; i<(n1-1); i++)
        {
        s = static_cast<double>(i)/(n1-1);
        if ( (1.0 - s - t) > 0.0 )
          {
          x[0] = x0[0] + s*(x1[0]-x0[0]) + t*(x2[0]-x0[0]);
          x[1] = x0[1] + s*(x1[1]-x0[1]) + t*(x2[1]-x0[1]);
          x[2] = x0[2] + s*(x1[2]-x0[2]) + t*(x2[2]-x0[2]);
          this->InsertNextPoint(output, x, inPts, 3, pts, s, t);
          }
        }
      }
    }
}
  
//------------------------------------------------------------------------
void vtkPolyDataPointSampler::SamplePolygon(vtkPolyData *output, vtkPoints *inPts, 
                                            vtkIdType npts, vtkIdType *pts)
{
  // Specialize for quads
  if ( npts == 4 )
    {
    double x0[3], x1[3], x2[3], x3[4];
    inPts->GetPoint(pts[0],x0);
    inPts->GetPoint(pts[1],x1);
    inPts->GetPoint(pts[2],x2);
    inPts->GetPoint(pts[3],x3);

    double l1 = vtkMath::Distance2BetweenPoints(x0,x1);
    double l2 = vtkMath::Distance2BetweenPoints(x0,x3);
    if ( l1 > this->Distance2 || l2 > this->Distance2 )
      {
      double s, t, x[3];
      l1 = sqrt(l1);
      l2 = sqrt(l2);
      int n1 = static_cast<int>(l1/this->Distance) + 2;
      int n2 = static_cast<int>(l2/this->Distance) + 2;
      n1 = ( n1 < 3 ? 3 : n1); //make sure there is at least one point
      n2 = ( n2 < 3 ? 3 : n2);
      for (vtkIdType j=1; j<(n2-1); j++)
        {
        t = static_cast<double>(j)/(n2-1);
        for (vtkIdType i=1; i<(n1-1); i++)
          {
          s = static_cast<double>(i)/(n1-1);
          x[0] = x0[0] + s*(x1[0]-x0[0]) + t*(x3[0]-x0[0]);
          x[1] = x0[1] + s*(x1[1]-x0[1]) + t*(x3[1]-x0[1]);
          x[2] = x0[2] + s*(x1[2]-x0[2]) + t*(x3[2]-x0[2]);
          this->InsertNextPoint(output, x, inPts, npts, pts, s ,t);
          }
        }
      }
    }
  else
    {
    }
}


//------------------------------------------------------------------------
void vtkPolyDataPointSampler
::InsertNextPoint(vtkPolyData *output, double x[3], vtkPoints *vtkNotUsed(inPts),
                  vtkIdType vtkNotUsed(npts), vtkIdType *vtkNotUsed(pts),
                  double vtkNotUsed(s), double vtkNotUsed(t))
{
  output->GetPoints()->InsertNextPoint(x);
}

//------------------------------------------------------------------------
void vtkPolyDataPointSampler::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << "Distance: " << this->Distance << "\n";

  os << indent << "Generate Vertex Points: " 
     << (this->GenerateVertexPoints ? "On\n" : "Off\n");
  os << indent << "Generate Edge Points: " 
     << (this->GenerateEdgePoints ? "On\n" : "Off\n");
  os << indent << "Generate Interior Points: " 
     << (this->GenerateInteriorPoints ? "On\n" : "Off\n");

  os << indent << "Generate Vertices: " 
     << (this->GenerateVertices ? "On\n" : "Off\n");
}