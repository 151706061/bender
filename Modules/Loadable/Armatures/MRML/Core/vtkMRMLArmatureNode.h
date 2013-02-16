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

#ifndef __vtkMRMLArmatureNode_h
#define __vtkMRMLArmatureNode_h

// Slicer includes
#include "vtkMRMLAnnotationHierarchyNode.h"

// Armatures includes
#include "vtkBenderArmaturesModuleMRMLCoreExport.h"
#include "vtkArmatureWidget.h"

class vtkArmatureWidget;
class vtkBoneRepresentation;
class vtkBoneWidget;
class vtkCallbackCommand;
class vtkMRMLArmatureDisplayableManager;
class vtkMRMLBoneNode;
class vtkMRMLModelNode;

// VTK includes
#include "vtkCommand.h"
class vtkPolyData;

/// \ingroup Bender_MRML
/// \brief Root of a tree of bones
///
/// \sa vtkMRMLBoneNode
class VTK_BENDER_ARMATURES_MRML_CORE_EXPORT vtkMRMLArmatureNode
  : public vtkMRMLAnnotationHierarchyNode
{
public:
  //--------------------------------------------------------------------------
  // VTK methods
  //--------------------------------------------------------------------------

  static vtkMRMLArmatureNode *New();
  vtkTypeMacro(vtkMRMLArmatureNode,vtkMRMLAnnotationHierarchyNode);
  virtual void PrintSelf(ostream& os, vtkIndent indent);

  //--------------------------------------------------------------------------
  // MRMLNode methods
  //--------------------------------------------------------------------------

  /// Instantiate a bone node.
  virtual vtkMRMLNode* CreateNodeInstance();

  /// Get node XML tag name (like Volume, Model).
  virtual const char* GetNodeTagName() {return "Armature";};

  /// Read node attributes from XML file.
  virtual void ReadXMLAttributes( const char** atts);

  /// Write this node's information to a MRML file in XML format.
  virtual void WriteXML(ostream& of, int indent);

  /// Copy the node's attributes to this object.
  virtual void Copy(vtkMRMLNode *node);

  /// Update references from scene.
  virtual void UpdateScene(vtkMRMLScene *scene);

  /// Alternative method to propagate events generated by observed nodes.
  virtual void ProcessMRMLEvents(vtkObject* caller,
                                 unsigned long event,
                                 void* callData);

  //--------------------------------------------------------------------------
  // Annotation methods
  //--------------------------------------------------------------------------

  virtual const char* GetIcon();

  //--------------------------------------------------------------------------
  // Armature methods
  //--------------------------------------------------------------------------

  /// Set/Get the bones representation type.
  /// \sa GetBonesRepresentationType(), SetBonesRepresentationType()
  void SetBonesRepresentation(vtkBoneRepresentation* rep);
  vtkBoneRepresentation* GetBonesRepresentation();

  /// Set/Get the bones representation type.
  /// \sa SetBonesRepresentation(), GetBonesRepresentation()
  void SetBonesRepresentationType(int representationType);
  int GetBonesRepresentationType();

  //BTX
  enum RepresentationTypes
    {
    Bone = 0,
    Cylinder,
    Octohedron
    };
  //ETX

  /// Set the bones widget state.
  /// \sa GetWidgetState()
  void SetWidgetState(int state);
  /// Get the bones widget state.
  /// \sa SetWidgetState()
  int GetWidgetState();

  //BTX
  enum WidgetState
    {
    Rest = vtkArmatureWidget::Rest,
    Pose = vtkArmatureWidget::Pose
    };
  //ETX

  /// Set the bones debug axes visibility.
  /// \sa GetShowAxes()
  void SetShowAxes(int axesVisibility);
  /// Get the bones debug axes visibility.
  /// \sa SetShowAxes()
  int GetShowAxes();

  /// The different axes that can be shown.
  //BTX
  enum ArmatureAxesType
    {
    Hidden = 0,
    ShowRestTransform,
    ShowPoseTransform,
    };
  //ETX

  /// Show the a line between the bones and their origin.
  /// \sa GetShowParenthood()
  void SetShowParenthood(int parenthood);
  /// Hide the a line between the bones and their origin.
  /// \sa SetShowParenthood()
  int GetShowParenthood();

  /// Show/Hide the armature.
  /// \sa GetVisibility(), SetVisibility()
  void SetVisibility(bool visible);
  bool GetVisibility();

  /// Set/Get the armature opacity.
  /// The armature opacity will be propagated to all its bones.
  /// \sa GetOpacity(), SetOpacity()
  void SetOpacity(double opacity);
  double GetOpacity();

  /// Set/Get the armature color.
  /// The armature color will be propagated to all its bones.
  /// \sa GetColor(), SetColor()
  void SetColor(double rgb[3]);
  void GetColor(double rgb[3]);

  /// Set/Get the armature display property bones always on top.
  /// \sa GetBonesAlwaysOnTop(), SetBonesAlwaysOnTop()
  void SetBonesAlwaysOnTop(int onTop);
  int GetBonesAlwaysOnTop();

  /*/// Add a widget bone to the armature.
  /// \sa RemoveBone()
  void AddBone(vtkBoneWidget* bone,
    vtkBoneWidget* boneParent = 0, bool linkedWithParent = true);*/

  /*/// Set/Get the bone is linked with its parents or not.
  /// \sa GetColor(), SetColor()
  void SetBoneLinkedWithParent(vtkBoneWidget* bone, bool linked);
  bool GetBoneLinkedWithParent(vtkBoneWidget* bone);*/

  //BTX
  enum MRMLArmatureNode
    {
    ArmatureBoneModified = vtkCommand::UserEvent + 1
    };
  //ETX

  /// Show/Hide the bone's envelopes.
  /// \sa SetShowEnvelopes(), GetShowEnvelopes()
  void SetShowEnvelopes(int show);
  int GetShowEnvelopes();

  /// Set/Get the overall radius ratio.
  vtkSetMacro(OverallRadiusRatio, double);
  vtkGetMacro(OverallRadiusRatio, double);

  /// Set/Get the envelopes opacity.
  /// The envelopes opacity will be propagated to all the bones' envelopes.
  void SetEnvelopeOpacity(double opacity);
  double GetEnvelopeOpacity();


  //--------------------------------------------------------------------------
  // Bone methods
  //--------------------------------------------------------------------------
  /// Search and populate the \a bones collection with all the bones found
  /// in the armature.
  /// \sa GetBoneParent()
  inline void GetAllBones(vtkCollection* bones);

  /// Search and return the parent of the \a boneNode bone. Return 0 if the
  /// bone doesn't have a parent (top-level) or doesn't belong to the
  /// armature.
  /// \sa GetAllBones()
  vtkMRMLBoneNode* GetParentBone(vtkMRMLBoneNode* boneNode);

  //--------------------------------------------------------------------------
  // Helper methods
  //--------------------------------------------------------------------------

  /// Copy the properties of the widget into the node
  /// \sa PasteArmatureNodeProperties()
  void CopyArmatureWidgetProperties(vtkArmatureWidget* armatureWidget);

  /// Paste the properties of the node into the widget
  /// \sa CopyArmatureWidgetProperties()
  void PasteArmatureNodeProperties(vtkArmatureWidget* armatureWidget);

  /*//--------------------------------------------------------------------------
  // Armature Node methods
  //--------------------------------------------------------------------------

  /// Get the armature node callback command. Used to listen to
  /// vtkArmatureWidget AddedBone and RemovedBone events.
  vtkMRMLArmatureNodeCallback* GetArmatureNodeCallbackCommand();*/

  /// Reset the pose mode to the rest positions.
  void ResetPoseMode();

  /// Set the model that contains the polydata of the armature.
  /// The model node is the associated node of the armature.
  /// \sa GetArmatureModel(), SetAssociatedNodeID()
  void SetArmatureModel(vtkMRMLModelNode* model);
  /// Return the associated model node.
  /// \sa SetArmatureModel(), GetPolyData()
  vtkMRMLModelNode* GetArmatureModel();

  /// Return the armature polydata. Each bone is represented by a 2-point
  /// line cell. It is the polydata stored in the associated model node.
  /// \sa SetPolyData(), GetArmatureModel()
  vtkPolyData* GetPolyData();
protected:
  vtkMRMLArmatureNode();
  ~vtkMRMLArmatureNode();

  vtkMRMLArmatureNode(const vtkMRMLArmatureNode&); /// not implemented
  void operator=(const vtkMRMLArmatureNode&); /// not implemented

  //BTX
  friend class vtkMRMLArmatureDisplayableManager;
  //ETX

  vtkCallbackCommand* Callback;

  /// Set the armature polydata. Only the displayable manager can set the
  /// polydata.
  /// \sa GetPolyData(), SetArmatureModel()
  void SetArmaturePolyData(vtkPolyData* polyData);

  /// Update the given bones representation to non-propagated property.
  /// For example, the opacity is not automatically propagated.
  void UpdateBoneRepresentation(vtkBoneRepresentation* rep);

  vtkArmatureWidget* ArmatureProperties;
  int WidgetState;
  int BonesRepresentationType;
  int ShouldResetPoseMode;
  double OverallRadiusRatio;
};

//----------------------------------------------------------------------------
void vtkMRMLArmatureNode::GetAllBones(vtkCollection* bones)
{
  this->GetAllChildren(bones);
}

#endif
