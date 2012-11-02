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

// Armatures includes
#include "vtkMRMLBoneNode.h"
#include "vtkMRMLBoneDisplayNode.h"
//#include "vtkMRMLBoneStorageNode.h"

// Bender includes
#include <vtkCylinderBoneRepresentation.h>
#include <vtkDoubleConeBoneRepresentation.h>
#include <vtkBoneRepresentation.h>
#include <vtkBoneWidget.h>

// MRML includes
#include <vtkMRMLScene.h>

// VTK includes
#include <vtkMath.h>
#include <vtkNew.h>
#include <vtkObjectFactory.h>

#include <vtkProperty.h>

// STD includes
//#include <algorithm>
//#include <cassert>
//#include <sstream>

//----------------------------------------------------------------------------
vtkMRMLNodeNewMacro(vtkMRMLBoneNode);

//----------------------------------------------------------------------------
class vtkMRMLBoneNodeCallback : public vtkCommand
{
public:
  static vtkMRMLBoneNodeCallback *New()
    { return new vtkMRMLBoneNodeCallback; }

  vtkMRMLBoneNodeCallback()
    { this->Node = 0; }

  virtual void Execute(vtkObject* caller, unsigned long eventId, void* data)
    {
    vtkNotUsed(data);
    switch (eventId)
      {
      case vtkCommand::ModifiedEvent:
        {
        this->Node->Modified();
        break;
        }
      }
    }

  vtkMRMLBoneNode* Node;
};

//----------------------------------------------------------------------------
vtkMRMLBoneNode::vtkMRMLBoneNode()
{
  this->Callback = vtkMRMLBoneNodeCallback::New();
  this->BoneProperties = vtkBoneWidget::New();
  this->BoneProperties->SetDebugBoneID(255);
  this->BoneRepresentationType = 0;

  this->Callback->Node = this;
  this->BoneProperties->AddObserver(vtkCommand::ModifiedEvent, this->Callback);

  this->BoneProperties->CreateDefaultRepresentation();
  this->BoneProperties->SetWidgetStateToRest();
}

//----------------------------------------------------------------------------
vtkMRMLBoneNode::~vtkMRMLBoneNode()
{
  this->Callback->Delete();
  this->BoneProperties->Delete();
}

//----------------------------------------------------------------------------
void vtkMRMLBoneNode::WriteXML(ostream& of, int nIndent)
{
  this->Superclass::WriteXML(of, nIndent);
  // of << indent << " ctrlPtsNumberingScheme=\"" << this->NumberingScheme << "\"";
}

//----------------------------------------------------------------------------
void vtkMRMLBoneNode::ReadXMLAttributes(const char** atts)
{
  int disabledModify = this->StartModify();

  this->Superclass::ReadXMLAttributes(atts);

  while (*atts != NULL)
    {
    //const char* attName = *(atts++);
    //std::string attValue(*(atts++));

    // if  (!strcmp(attName, "ctrlPtsNumberingScheme"))
    //   {
    //   std::stringstream ss;
    //   ss << attValue;
    //   ss >> this->NumberingScheme;
    //   }
    }
  this->EndModify(disabledModify);
}

//-----------------------------------------------------------
void vtkMRMLBoneNode::UpdateScene(vtkMRMLScene *scene)
{
  this->Superclass::UpdateScene(scene);
}

//---------------------------------------------------------------------------
void vtkMRMLBoneNode::ProcessMRMLEvents(vtkObject* caller,
                                        unsigned long event,
                                        void* callData)
{
  this->Superclass::ProcessMRMLEvents(caller, event, callData);
}

//----------------------------------------------------------------------------
void vtkMRMLBoneNode::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}

//----------------------------------------------------------------------------
void vtkMRMLBoneNode::Initialize(vtkMRMLScene* mrmlScene)
{
  if (!mrmlScene)
    {
    vtkErrorMacro(<< __FUNCTION__ << ": No scene");
    return;
    }
  // \tbd remove this SetScene call as it shouldn't be mandatory.
  this->SetScene(mrmlScene);
  this->Superclass::Initialize(mrmlScene);
  this->CreateBoneDisplayNode();
}

//----------------------------------------------------------------------------
vtkMRMLBoneDisplayNode* vtkMRMLBoneNode::GetBoneDisplayNode()
{
  return vtkMRMLBoneDisplayNode::SafeDownCast(
    this->GetNthDisplayNodeByClass(0, "vtkMRMLBoneDisplayNode"));
}

//---------------------------------------------------------------------------
void vtkMRMLBoneNode::CreateBoneDisplayNode()
{
  if (this->GetBoneDisplayNode() != 0)
    {
    return;
    }
  if (!this->GetScene())
    {
    vtkErrorMacro( << __FUNCTION__ << ": No scene" ) ;
    return;
    }

  vtkNew<vtkMRMLBoneDisplayNode> boneDisplayNode;
  this->GetScene()->AddNode(boneDisplayNode.GetPointer());

  this->AddAndObserveDisplayNodeID(boneDisplayNode->GetID());
}

//-- Name -------------------------------------------------------------------
//---------------------------------------------------------------------------
void vtkMRMLBoneNode::SetBoneName(vtkStdString name)
{
  this->BoneProperties->SetName(name);
  this->SetName(name);
}

//---------------------------------------------------------------------------
vtkStdString vtkMRMLBoneNode::GetBoneName()
{
  return this->BoneProperties->GetName();
}

//-- State ------------------------------------------------------------------
//---------------------------------------------------------------------------
void vtkMRMLBoneNode::SetWidgetState(int state)
{
  //this->BoneProperties->SetWidgetState(state); // \TO DO
}

//---------------------------------------------------------------------------
int vtkMRMLBoneNode::GetWidgetState()
{
  return this->BoneProperties->GetWidgetState();
}

//-- Representation ---------------------------------------------------------
namespace
{
int FindBoneRepresentationType(vtkBoneRepresentation* rep)
{
  if (vtkCylinderBoneRepresentation::SafeDownCast(rep))
    {
    return 1;
    }
  else if (vtkDoubleConeBoneRepresentation::SafeDownCast(rep))
    {
    return 2;
    }
  else
    {
    return 0;
    }
}
}// end namespace

//---------------------------------------------------------------------------
void vtkMRMLBoneNode::SetBoneRepresentation(vtkBoneRepresentation* r)
{
  this->SetBoneRepresentationType( FindBoneRepresentationType(r) );
}

//---------------------------------------------------------------------------
vtkBoneRepresentation* vtkMRMLBoneNode::GetBoneRepresentation()
{
  return this->BoneProperties->GetBoneRepresentation();
}

//---------------------------------------------------------------------------
void vtkMRMLBoneNode::SetBoneRepresentationType(int type)
{
  if (type == 1)
    {
    vtkCylinderBoneRepresentation* rep
      = vtkCylinderBoneRepresentation::New();
    this->BoneProperties->SetRepresentation(rep);
    this->BoneRepresentationType = 1;
    }
  else if (type == 2)
    {
    vtkDoubleConeBoneRepresentation* rep
      = vtkDoubleConeBoneRepresentation::New();
    this->BoneProperties->SetRepresentation(rep);
    this->BoneRepresentationType = 2;
    }
  else
    {
    vtkBoneRepresentation* rep
      = vtkBoneRepresentation::New();
    this->BoneProperties->SetRepresentation(rep);
    this->BoneRepresentationType = 0;
    }

  this->Modified();
}

//-- Color -------------------------------------------------------------------
//---------------------------------------------------------------------------
void vtkMRMLBoneNode::SetBoneColor(int rgb[3])
{
  double doubleRGB[3];
  doubleRGB[0] = rgb[0]/255.0;
  doubleRGB[1] = rgb[1]/255.0;
  doubleRGB[2] = rgb[2]/255.0;

  this->GetBoneRepresentation()->GetLineProperty()->SetColor(doubleRGB);
  this->Modified();
}

//---------------------------------------------------------------------------
void vtkMRMLBoneNode::GetBoneColor(int rgb[3])
{
  double doubleRGB[3];
  this->GetBoneRepresentation()->GetLineProperty()->GetColor(doubleRGB);
  rgb[0] = static_cast<int>(doubleRGB[0] * 255);
  rgb[1] = static_cast<int>(doubleRGB[1] * 255);
  rgb[2] = static_cast<int>(doubleRGB[2] * 255);
}

//-- Opacity ----------------------------------------------------------------
//---------------------------------------------------------------------------
void vtkMRMLBoneNode::SetOpacity(double opacity)
{
  if (this->BoneRepresentationType == 1)
    {
    vtkCylinderBoneRepresentation* rep
      = vtkCylinderBoneRepresentation::SafeDownCast(this->GetBoneRepresentation());
    rep->GetCylinderProperty()->SetOpacity(opacity);
    }
  else if (this->BoneRepresentationType  == 2)
    {
    vtkDoubleConeBoneRepresentation* rep
      = vtkDoubleConeBoneRepresentation::SafeDownCast(this->GetBoneRepresentation());
    rep->GetConesProperty()->SetOpacity(opacity);
    }

  this->GetBoneRepresentation()->GetLineProperty()->SetOpacity(opacity);
  this->Modified();
}

//---------------------------------------------------------------------------
double vtkMRMLBoneNode::GetOpacity()
{
  return this->GetBoneRepresentation()->GetLineProperty()->GetOpacity();
}

//-- Distance ---------------------------------------------------------------
//---------------------------------------------------------------------------
double vtkMRMLBoneNode::GetDistance()
{
  if (this->BoneProperties->GetWidgetState() == vtkBoneWidget::PlaceHead
    || this->BoneProperties->GetWidgetState() == vtkBoneWidget::PlaceTail)
    {
    return 0.0;
    }

  // Can't use bone properties -> rep -> distance because the representation
  // is never built. Thus the distance is never recomputed
  return sqrt(vtkMath::Distance2BetweenPoints(
    this->BoneProperties->GetBoneRepresentation()->GetWorldHeadPosition(),
    this->BoneProperties->GetBoneRepresentation()->GetWorldTailPosition()));
}

//-- Roll -------------------------------------------------------------------
//---------------------------------------------------------------------------
void vtkMRMLBoneNode::SetRoll(double roll)
{
  this->BoneProperties->SetRoll(roll);
}

//---------------------------------------------------------------------------
double vtkMRMLBoneNode::GetRoll()
{
  return this->BoneProperties->GetRoll();
}

//-- World Positions -------------------------------------------------------
//---------------------------------------------------------------------------
void vtkMRMLBoneNode::SetWorldHeadRest(const double* headPoint)
{
  double head[3];
  head[0] = headPoint[0];
  head[1] = headPoint[1];
  head[2] = headPoint[2];
  this->BoneProperties->SetWorldHeadRest(head);
}

//---------------------------------------------------------------------------
double* vtkMRMLBoneNode::GetWorldHeadRest()
{
  return this->BoneProperties->GetWorldHeadRest();
}

//---------------------------------------------------------------------------
double* vtkMRMLBoneNode::GetWorldHeadPose()
{
  return this->BoneProperties->GetWorldHeadPose();
}

//---------------------------------------------------------------------------
void vtkMRMLBoneNode::SetWorldTailRest(const double* tailPoint)
{
  double tail[3];
  tail[0] = tailPoint[0];
  tail[1] = tailPoint[1];
  tail[2] = tailPoint[2];
  this->BoneProperties->SetWorldTailRest(tail);
}

//---------------------------------------------------------------------------
double* vtkMRMLBoneNode::GetWorldTailRest()
{
  return this->BoneProperties->GetWorldTailRest();
}

//---------------------------------------------------------------------------
double* vtkMRMLBoneNode::GetWorldTailPose()
{
  return this->BoneProperties->GetWorldTailPose();
}

//-- Local Positions -------------------------------------------------------
//---------------------------------------------------------------------------
void vtkMRMLBoneNode::SetLocalHeadRest(const double* headPoint)
{
  double head[3];
  head[0] = headPoint[0];
  head[1] = headPoint[1];
  head[2] = headPoint[2];
  this->BoneProperties->SetLocalHeadRest(head);
}

//---------------------------------------------------------------------------
void vtkMRMLBoneNode::SetLocalTailRest(const double* tailPoint)
{
  double tail[3];
  tail[0] = tailPoint[0];
  tail[1] = tailPoint[1];
  tail[2] = tailPoint[2];
  this->BoneProperties->SetLocalTailRest(tail);
}

//---------------------------------------------------------------------------
double* vtkMRMLBoneNode::GetLocalHeadRest()
{
  return this->BoneProperties->GetLocalHeadRest();
}

//---------------------------------------------------------------------------
double* vtkMRMLBoneNode::GetLocalHeadPose()
{
  return this->BoneProperties->GetLocalHeadPose();
}

//---------------------------------------------------------------------------
double* vtkMRMLBoneNode::GetLocalTailRest()
{
  return this->BoneProperties->GetLocalTailRest();
}

//---------------------------------------------------------------------------
double* vtkMRMLBoneNode::GetLocalTailPose()
{
  return this->BoneProperties->GetLocalTailPose();
}

//-- Axes Visibility ---------------------------------------------------------
void vtkMRMLBoneNode::SetAxesVisibility(int axesVisibility)
{
  this->BoneProperties->SetAxesVisibility(axesVisibility);
}

//---------------------------------------------------------------------------
int vtkMRMLBoneNode::GetAxesVisibility()
{
  return this->BoneProperties->GetAxesVisibility();
}

//-- World to parent transforms ---------------------------------------------
//---------------------------------------------------------------------------
void vtkMRMLBoneNode::SetWorldToParentRestRotation(const double* rotation)
{
  double rot[4];
  rot[0] = rotation[0];
  rot[1] = rotation[1];
  rot[2] = rotation[2];
  rot[3] = rotation[3];
  this->BoneProperties->SetWorldToParentRestRotation(rot);
}

//---------------------------------------------------------------------------
void vtkMRMLBoneNode::SetWorldToParentPoseRotation(const double* rotation)
{
  double rot[4];
  rot[0] = rotation[0];
  rot[1] = rotation[1];
  rot[2] = rotation[2];
  rot[3] = rotation[3];
  this->BoneProperties->SetWorldToParentPoseRotation(rot);
}

//---------------------------------------------------------------------------
double* vtkMRMLBoneNode::GetWorldToParentRestRotation()
{
  return this->BoneProperties->GetWorldToParentRestRotation();
}

//---------------------------------------------------------------------------
double* vtkMRMLBoneNode::GetWorldToParentPoseRotation()
{
  return this->BoneProperties->GetWorldToParentPoseRotation();
}

//---------------------------------------------------------------------------
void vtkMRMLBoneNode::SetWorldToParentRestTranslation(const double* translation)
{
  double trans[3];
  trans[0] = translation[0];
  trans[1] = translation[1];
  trans[2] = translation[2];
  this->BoneProperties->SetWorldToParentRestTranslation(trans);
}

//---------------------------------------------------------------------------
void vtkMRMLBoneNode::SetWorldToParentPoseTranslation(const double* translation)
{
  double trans[3];
  trans[0] = translation[0];
  trans[1] = translation[1];
  trans[2] = translation[2];
  this->BoneProperties->SetWorldToParentPoseTranslation(trans);
}

//---------------------------------------------------------------------------
double* vtkMRMLBoneNode::GetWorldToParentRestTranslation()
{
  return this->BoneProperties->GetWorldToParentRestTranslation();
}

//---------------------------------------------------------------------------
double* vtkMRMLBoneNode::GetWorldToParentPoseTranslation()
{
  return this->BoneProperties->GetWorldToParentPoseTranslation();
}

//-- Parent to bone transforms ---------------------------------------------
//---------------------------------------------------------------------------
double* vtkMRMLBoneNode::GetParentToBoneRestRotation()
{
  return this->BoneProperties->GetParentToBoneRestRotation();
}

//---------------------------------------------------------------------------
double* vtkMRMLBoneNode::GetParentToBonePoseRotation()
{
  return this->BoneProperties->GetParentToBonePoseRotation();
}

//---------------------------------------------------------------------------
double* vtkMRMLBoneNode::GetParentToBoneRestTranslation()
{
  return this->BoneProperties->GetParentToBoneRestTranslation();
}

//---------------------------------------------------------------------------
double* vtkMRMLBoneNode::GetParentToBonePoseTranslation()
{
  return this->BoneProperties->GetParentToBonePoseTranslation();
}

//-- World to bone transforms -----------------------------------------------
//---------------------------------------------------------------------------
double* vtkMRMLBoneNode::GetWorldToBoneRestRotation()
{
  return this->BoneProperties->GetWorldToBoneRestRotation();
}

//---------------------------------------------------------------------------
double* vtkMRMLBoneNode::GetWorldToBonePoseRotation()
{
  return this->BoneProperties->GetWorldToBonePoseRotation();
}

//---------------------------------------------------------------------------
double* vtkMRMLBoneNode::GetWorldToBoneHeadRestTranslation()
{
  return this->BoneProperties->GetWorldToBoneHeadRestTranslation();
}

//---------------------------------------------------------------------------
double* vtkMRMLBoneNode::GetWorldToBoneHeadPoseTranslation()
{
  return this->BoneProperties->GetWorldToBoneHeadPoseTranslation();
}

//---------------------------------------------------------------------------
double* vtkMRMLBoneNode::GetWorldToBoneTailRestTranslation()
{
  return this->BoneProperties->GetWorldToBoneTailRestTranslation();
}

//---------------------------------------------------------------------------
double* vtkMRMLBoneNode::GetWorldToBoneTailPoseTranslation()
{
  return this->BoneProperties->GetWorldToBoneTailPoseTranslation();
}

//-- Parenthood -------------------------------------------------------------
//---------------------------------------------------------------------------
void vtkMRMLBoneNode::SetShowParenthood(int parenthood)
{
  this->BoneProperties->SetShowParenthood(parenthood);
}

//---------------------------------------------------------------------------
int vtkMRMLBoneNode::GetShowParenthood()
{
  return this->BoneProperties->GetShowParenthood();
}

//---------------------------------------------------------------------------
void vtkMRMLBoneNode::CopyBoneWidgetProperties(vtkBoneWidget* boneWidget)
{
  // -- World coordinates --
  this->BoneProperties->SetWorldHeadRest(boneWidget->GetWorldHeadRest());
  this->BoneProperties->SetWorldTailRest(boneWidget->GetWorldTailRest());

  // -- Local coordinates ==
  this->BoneProperties->SetLocalHeadRest(boneWidget->GetLocalHeadRest());
  this->BoneProperties->SetLocalTailRest(boneWidget->GetLocalTailRest());

  // -- World to parent transforms
  this->BoneProperties->SetWorldToParentRestRotation(
    boneWidget->GetWorldToParentRestRotation());
  this->BoneProperties->SetWorldToParentPoseRotation(
    boneWidget->GetWorldToParentPoseRotation());
  this->BoneProperties->SetWorldToParentRestTranslation(
    boneWidget->GetWorldToParentRestTranslation());
  this->BoneProperties->SetWorldToParentPoseTranslation(
    boneWidget->GetWorldToParentPoseTranslation());

  // -- Axes -- 
  this->BoneProperties->SetAxesVisibility(boneWidget->GetAxesVisibility());

  // -- Parenthood -- 
  this->BoneProperties->SetShowParenthood(boneWidget->GetShowParenthood());

  // -- Representation --
  this->SetBoneRepresentation(boneWidget->GetBoneRepresentation());

  this->BoneProperties->GetBoneRepresentation()->GetLineProperty()->SetColor(
    boneWidget->GetBoneRepresentation()->GetLineProperty()->GetColor());
  this->BoneProperties->GetBoneRepresentation()->GetLineProperty()->SetOpacity(
    boneWidget->GetBoneRepresentation()->GetLineProperty()->GetOpacity());

  // -- State --
  //this->BoneProperties->SetWidgetState(boneWidget->GetWidgeState()); // \TO DO

  // -- Name --
  this->SetBoneName(boneWidget->GetName());

  this->Modified();
}

//---------------------------------------------------------------------------
void vtkMRMLBoneNode::PasteBoneNodeProperties(vtkBoneWidget* boneWidget)
{
  // -- Representation, part 1 --
  if (FindBoneRepresentationType(boneWidget->GetBoneRepresentation()) != 
      this->BoneRepresentationType)
    {
    if (this->BoneRepresentationType == 1)
      {
      vtkCylinderBoneRepresentation* rep
        = vtkCylinderBoneRepresentation::New();
      boneWidget->SetRepresentation(rep);
      }
    else if (this->BoneRepresentationType == 2)
      {
      vtkDoubleConeBoneRepresentation* rep
        = vtkDoubleConeBoneRepresentation::New();
      boneWidget->SetRepresentation(rep);
      }
    else
      {
      vtkBoneRepresentation* rep = vtkBoneRepresentation::New();
      boneWidget->SetRepresentation(rep);
      }
    }

  // -- World coordinates --
  boneWidget->SetWorldHeadRest(this->BoneProperties->GetWorldHeadRest());
  boneWidget->SetWorldTailRest(this->BoneProperties->GetWorldTailRest());

  // -- Local coordinates ==
  boneWidget->SetLocalHeadRest(this->BoneProperties->GetLocalHeadRest());
  boneWidget->SetLocalTailRest(this->BoneProperties->GetLocalTailRest());

  // -- World to parent transforms
  boneWidget->SetWorldToParentRestRotation(
    this->BoneProperties->GetWorldToParentRestRotation());
  boneWidget->SetWorldToParentPoseRotation(
    this->BoneProperties->GetWorldToParentPoseRotation());
  boneWidget->SetWorldToParentRestTranslation(
    this->BoneProperties->GetWorldToParentRestTranslation());
  boneWidget->SetWorldToParentPoseTranslation(
    this->BoneProperties->GetWorldToParentPoseTranslation());

  // -- Axes -- 
  boneWidget->SetAxesVisibility(this->BoneProperties->GetAxesVisibility());

  // -- Parenthood -- 
  boneWidget->SetShowParenthood(this->BoneProperties->GetShowParenthood());

  // -- Representation, part 2 --
  double rgb[3];
  this->GetBoneRepresentation()->GetLineProperty()->GetColor(rgb);
  if (this->BoneRepresentationType == 1)
    {
    vtkCylinderBoneRepresentation* rep
      = vtkCylinderBoneRepresentation::SafeDownCast(
        boneWidget->GetBoneRepresentation());
    rep->GetCylinderProperty()->SetColor(rgb);
    rep->GetCylinderProperty()->SetOpacity(this->GetOpacity());
    }
  else if (this->BoneRepresentationType == 2)
    {
    vtkDoubleConeBoneRepresentation* rep
      = vtkDoubleConeBoneRepresentation::SafeDownCast(
        boneWidget->GetBoneRepresentation());
    rep->GetConesProperty()->SetColor(rgb);
    rep->GetConesProperty()->SetOpacity(this->GetOpacity());
    }

  //std::cout<<"RGB: "<<rgb[0]<<" "<<rgb[1]<<" "<<rgb[2]<<std::endl;

  boneWidget->GetBoneRepresentation()->GetLineProperty()->SetColor(rgb);
  boneWidget->GetBoneRepresentation()->GetLineProperty()->SetOpacity(this->GetOpacity());

  // -- State --
  //boneWidget->SetWidgetState(this->BoneProperties->GetWidgeState()); // \TO DO

  // -- Name --
  boneWidget->SetName(this->BoneProperties->GetName());
}
