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

#include "benderIOUtils.h"

// VTK includes
#include <vtkCellArray.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyDataWriter.h>
#include <vtkSTLReader.h>
#include <vtkSmartPointer.h>
#include <vtkXMLPolyDataReader.h>

// ITK includes
#include <itkMesh.h>
#include <itkLineCell.h>
#include <itkTriangleCell.h>

namespace bender
{
//-----------------------------------------------------------------------------
vtkPolyData* IOUtils::ReadPolyData(const std::string& fileName, bool invertXY)
{
  vtkPolyData* polyData = 0;
  vtkSmartPointer<vtkPolyDataReader> pdReader;
  vtkSmartPointer<vtkXMLPolyDataReader> pdxReader;
  vtkSmartPointer<vtkSTLReader> stlReader;

  // do we have vtk or vtp models?
  std::string::size_type loc = fileName.find_last_of(".");
  if( loc == std::string::npos )
    {
    std::cerr << "Failed to find an extension for " << fileName << std::endl;
    return polyData;
    }

  std::string extension = fileName.substr(loc);

  if( extension == std::string(".vtk") )
    {
    pdReader = vtkSmartPointer<vtkPolyDataReader>::New();
    pdReader->SetFileName(fileName.c_str() );
    pdReader->Update();
    polyData = pdReader->GetOutput();
    }
  else if( extension == std::string(".vtp") )
    {
    pdxReader = vtkSmartPointer<vtkXMLPolyDataReader>::New();
    pdxReader->SetFileName(fileName.c_str() );
    pdxReader->Update();
    polyData = pdxReader->GetOutput();
    }
  else if( extension == std::string(".stl") )
    {
    stlReader = vtkSmartPointer<vtkSTLReader>::New();
    stlReader->SetFileName(fileName.c_str() );
    stlReader->Update();
    polyData = stlReader->GetOutput();
    }
  if( polyData == NULL )
    {
    std::cerr << "Failed to read surface " << fileName << std::endl;
    return polyData;
    }
  // Build Cells
  polyData->BuildLinks();

  if(invertXY)
    {
    vtkPoints* points = polyData->GetPoints();
    for(int i=0; i<points->GetNumberOfPoints();++i)
      {
      double x[3];
      points->GetPoint(i,x);
      x[0]*=-1;
      x[1]*=-1;
      points->SetPoint(i, x);
      }
    }
  polyData->Register(0);
  return polyData;
}

//-----------------------------------------------------------------------------
void IOUtils::WritePolyData(vtkPolyData* polyData, const std::string& fileName)
{
  cout<<"Write polydata to "<<fileName<<endl;
  vtkNew<vtkPolyDataWriter> pdWriter;
  pdWriter->SetInput(polyData);
  pdWriter->SetFileName(fileName.c_str() );
  pdWriter->SetFileTypeToBinary();
  pdWriter->Update();
}
//-----------------------------------------------------------------------------
itk::Mesh<double, 3, IOUtils::MeshTraits>::Pointer IOUtils
::VTKPolyDataToITKMesh(vtkPolyData* polyData)
{
  typedef itk::Mesh<double, 3, MeshTraits> MeshType;
  MeshType::Pointer  mesh = MeshType::New();

  //
  // Transfer the points from the vtkPolyData into the itk::Mesh
  //
  const unsigned int numberOfPoints = polyData->GetNumberOfPoints();

  vtkPoints * vtkpoints = polyData->GetPoints();

  mesh->GetPoints()->Reserve( numberOfPoints );

  for(unsigned int p =0; p < numberOfPoints; p++)
    {
    double * apoint = vtkpoints->GetPoint( p );
    mesh->SetPoint( p, MeshType::PointType( apoint ));
    }

  //
  // Transfer the cells from the vtkPolyData into the itk::Mesh
  //
  vtkCellArray * triangleStrips = polyData->GetStrips();


  vtkIdType  * cellPoints;
  vtkIdType    numberOfCellPoints;

  //
  // First count the total number of triangles from all the triangle strips.
  //
  unsigned int numberOfTriangles = 0;

  triangleStrips->InitTraversal();

  while( triangleStrips->GetNextCell( numberOfCellPoints, cellPoints ) )
    {
    numberOfTriangles += numberOfCellPoints-2;
    }


  vtkCellArray * polygons = polyData->GetPolys();

  polygons->InitTraversal();

  while( polygons->GetNextCell( numberOfCellPoints, cellPoints ) )
    {
    if( numberOfCellPoints == 3 )
      {
      numberOfTriangles ++;
      }
    }

  //
  // Reserve memory in the itk::Mesh for all those triangles
  //
  mesh->GetCells()->Reserve( numberOfTriangles );

  //
  // Copy the triangles from vtkPolyData into the itk::Mesh
  //
  //

  typedef MeshType::CellType   CellType;
  typedef itk::TriangleCell< CellType > TriangleCellType;

  int cellId = 0;

  // first copy the triangle strips
  triangleStrips->InitTraversal();
  while( triangleStrips->GetNextCell( numberOfCellPoints, cellPoints ) )
    {
    unsigned int numberOfTrianglesInStrip = numberOfCellPoints - 2;

    unsigned long pointIds[3];
    pointIds[0] = cellPoints[0];
    pointIds[1] = cellPoints[1];
    pointIds[2] = cellPoints[2];

    for( unsigned int t=0; t < numberOfTrianglesInStrip; t++ )
      {
      MeshType::CellAutoPointer c;
      TriangleCellType * tcell = new TriangleCellType;
      tcell->SetPointIds( pointIds );
      c.TakeOwnership( tcell );
      mesh->SetCell( cellId, c );
      cellId++;
      pointIds[0] = pointIds[1];
      pointIds[1] = pointIds[2];
      pointIds[2] = cellPoints[t+3];
      }
    }


  // then copy the normal triangles
  polygons->InitTraversal();
  while( polygons->GetNextCell( numberOfCellPoints, cellPoints ) )
    {
    if( numberOfCellPoints !=3 ) // skip any non-triangle.
      {
      continue;
      }
    MeshType::CellAutoPointer c;
    TriangleCellType * t = new TriangleCellType;
    t->SetPointIds( (unsigned long*)cellPoints );
    c.TakeOwnership( t );
    mesh->SetCell( cellId, c );
    cellId++;
    }


  std::cout << "Mesh  " << std::endl;
  std::cout << "Number of Points =   " << mesh->GetNumberOfPoints() << std::endl;
  std::cout << "Number of Cells  =   " << mesh->GetNumberOfCells()  << std::endl;

  return mesh;
}

//-----------------------------------------------------------------------------
void IOUtils::FilterStart(const char* filterName, const char* comment)
{
  assert(filterName);
  std::cout << "<filter-start>\n";
  std::cout << "<filter-name>"
            << filterName
            << "</filter-name>\n";
  if (comment)
    {
    std::cout << "filter-comment>"
              << comment
              << "</filter-comment>\n";
    }
  std::cout << "</filter-start>\n";
  std::cout << std::flush;
}

//-----------------------------------------------------------------------------
void IOUtils::FilterProgress(const char* filterName, float progress,
                             double fraction, double start)
{
  assert(filterName);
  assert(progress >= 0. && progress <= 1.0);
  assert(fraction > 0. && fraction <= 1.0 );
  assert(start >= 0. && start < 1.0);
  std::cout << "<filter-progress>"
            << start + (progress * fraction)
            << "</filter-progress>\n";
  if (fraction != 1.0)
    {
    std::cout << "<filter-stage-progress>"
              << progress
              << "</filter-stage-progress>\n";
    }
  std::cout << std::flush;
}

//-----------------------------------------------------------------------------
void IOUtils::FilterEnd(const char* filterName, size_t time)
{
  assert(filterName);
  std::cout << "<filter-end>\n";
  std::cout << "<filter-name>"
            << filterName
            << "</filter-name>\n";
  // Not supported in old slicer versions
  //std::cout << "  <filter-time>"
  //          << time
  //          << "</filter-time>\n";
  std::cout << "</filter-end>\n";
  std::cout << std::flush;
}


};
