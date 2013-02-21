from __main__ import vtk, qt, ctk, slicer

#
# Workflow
#

class Workflow:
  def __init__(self, parent):
    import string
    parent.title = "Bender Workflow"
    parent.categories = ["", "Segmentation.Bender"]
    parent.contributors = ["Julien Finet (Kitware), Johan Andruejol (Kitware)"]
    parent.helpText = string.Template("""
Use this module to repose a humanoid labelmap. See <a href=\"$a/Documentation/$b.$c/Modules/Workflow\">$a/Documentation/$b.$c/Modules/Workflow</a> for more information.
    """).substitute({ 'a':'http://public.kitware.com/Wiki/Bender', 'b':0, 'c':1 })
    parent.acknowledgementText = """
    This work is supported by Air Force Research Laboratory (AFRL)
    """
    self.parent = parent

#
# Workflow widget
#

class WorkflowWidget:
  def __init__(self, parent = None):
    if not parent:
      self.setup()
      self.parent.show()
    else:
      self.parent = parent
    self.logic = None
    self.labelmapNode = None
    self.parent.show()

  def setup(self):
    import imp, sys, os, slicer
    loader = qt.QUiLoader()
    moduleName = 'Workflow'
    scriptedModulesPath = eval('slicer.modules.%s.path' % moduleName.lower())
    scriptedModulesPath = os.path.dirname(scriptedModulesPath)
    path = os.path.join(scriptedModulesPath, 'Resources', 'UI', 'Workflow.ui')

    qfile = qt.QFile(path)
    qfile.open(qt.QFile.ReadOnly)
    widget = loader.load( qfile, self.parent )
    self.layout = self.parent.layout()
    self.widget = widget;
    self.layout.addWidget(widget)

    self.reloadButton = qt.QPushButton("Reload")
    self.reloadButton.toolTip = "Reload this module."
    self.reloadButton.name = "Workflow Reload"
    self.layout.addWidget(self.reloadButton)
    self.reloadButton.connect('clicked()', self.reloadModule)

    self.WorkflowWidget = self.get('WorkflowWidget')
    self.TitleLabel = self.get('TitleLabel')

    # Labelmap variables
    self.oldLabelMapVolumeNode = None
    self.volumeNodeLabelMapTag = 0
    self.labelmapDisplayNodeLabelMapTag = 0

    # Transform variables
    self.TransformNode = None

    # Merge variables
    self.volumeNodeMergeLabelsTag = 0
    self.labelmapDisplayNodeMergeLabelsTag = 0

    # Pose Body variables
    self.PoseBodyCLI = slicer.mrmlScene.GetFirstNodeByName("Pose Body")
    if self.PoseBodyCLI == None:
      self.PoseBodyCLI = slicer.cli.createNode(slicer.modules.posebody)
      self.PoseBodyCLI.SetName("Pose Body")

    self.PoseBodyCLI.AddObserver('ParameterChangedEvent', self.updatePoseBodyFromCLI)

    # --------------------------------------------------------------------------
    # Connections
    # Workflow
    self.get('NextPageToolButton').connect('clicked()', self.goToNext)
    self.get('PreviousPageToolButton').connect('clicked()', self.goToPrevious)
    # 0) Welcome
    self.get('WelcomeSimpleWorkflowCheckBox').connect('stateChanged(int)', self.setupSimpleWorkflow)
    # 1) Bone Segmentations
    # a) Labelmap
    self.get('LabelmapVolumeNodeComboBox').connect('currentNodeChanged(vtkMRMLNode*)', self.setupLabelmap)
    self.get('LabelMapApplyColorNodePushButton').connect('clicked()', self.applyColorNode)
    self.get('LabelmapGoToModulePushButton').connect('clicked()', self.openLabelmapModule)
    self.get('TransformApplyPushButton').connect('clicked()', self.runTransform)
    # c) Merge Labels
    self.get('MergeLabelsInputNodeComboBox').connect('currentNodeChanged(vtkMRMLNode*)', self.setupMergeLabels)
    self.get('MergeLabelsApplyPushButton').connect('clicked()', self.runMergeLabels)
    self.get('MergeLabelsGoToModulePushButton').connect('clicked()', self.openMergeLabelsModule)
    # 2) Model Maker
    # a) Bone Model Maker
    self.get('BoneLabelComboBox').connect('currentColorChanged(int)', self.setupBoneModelMakerLabels)
    self.get('BoneModelMakerApplyPushButton').connect('clicked()', self.runBoneModelMaker)
    self.get('BoneModelMakerGoToModulePushButton').connect('clicked()', self.openBoneModelMakerModule)
    # b) Skin Model Maker
    self.get('SkinModelMakerInputNodeComboBox').connect('currentNodeChanged(vtkMRMLNode*)', self.setupSkinModelMakerLabels)
    self.get('SkinModelMakerToggleVisiblePushButtton').connect('clicked()', self.updateSkinNodeVisibility)
    self.get('SkinModelMakerApplyPushButton').connect('clicked()', self.runSkinModelMaker)
    self.get('SkinModelMakerGoToModulePushButton').connect('clicked()', self.openSkinModelMakerModule)
    # c) Volume Render
    self.get('BoneLabelComboBox').connect('currentColorChanged(int)', self.setupVolumeRenderLabels)
    self.get('SkinLabelComboBox').connect('currentColorChanged(int)', self.setupVolumeRenderLabels)
    self.get('VolumeRenderInputNodeComboBox').connect('currentNodeChanged(vtkMRMLNode*)', self.setupVolumeRender)
    self.get('VolumeRenderLabelsLineEdit').connect('editingFinished()', self.updateVolumeRenderLabels)
    self.get('VolumeRenderCheckBox').connect('toggled(bool)',self.runVolumeRender)
    self.get('VolumeRenderGoToModulePushButton').connect('clicked()', self.openVolumeRenderModule)
    # 3) Armatures
    self.get('ArmaturesGoToPushButton').connect('clicked()', self.openArmaturesModule)
    # 4) Armature Weight and Bones
    # a) Armatures Bones
    self.get('SegmentBonesApplyPushButton').connect('clicked()',self.runSegmentBones)
    self.get('SegmentBonesGoToPushButton').connect('clicked()', self.openSegmentBonesModule)
    # b) Armatures Weight
    self.get('ArmatureWeightApplyPushButton').connect('clicked()',self.runArmatureWeight)
    self.get('ArmatureWeightGoToPushButton').connect('clicked()', self.openArmatureWeightModule)
    # 5) (Pose) Armature And Pose Body
    # a) (Pose) Armatures
    self.get('PoseArmaturesGoToPushButton').connect('clicked()', self.openPosedArmatureModule)
    # b) Pose Body
    self.get('PoseBodySurfaceOutputNodeComboBox').connect('currentNodeChanged(vtkMRMLNode*)', self.setupPoseBody)
    self.get('PoseBodyWeightInputDirectoryButton').connect('directoryChanged(QString)', self.setupPoseBody)
    self.get('PoseBodySurfaceInputComboBox').connect('currentNodeChanged(vtkMRMLNode*)', self.setupPoseBody)
    self.get('PoseBodyArmatureInputNodeComboBox').connect('currentNodeChanged(vtkMRMLNode*)', self.setupPoseBody)
    self.get('PoseBodyApplyPushButton').connect('clicked()', self.runPoseBody)

    self.get('PoseBodyGoToPushButton').connect('clicked()', self.openPoseBodyModule)
    self.get('ArmatureWeightOutputDirectoryButton').connect('directoryChanged(QString)', self.setWeightDirectory)
    # 6) Resample
    self.get('ResampleApplyPushButton').connect('clicked()', self.runResample)

    # --------------------------------------------------------------------------
    # Initialize
    self.widget.setMRMLScene(slicer.mrmlScene)

    # Init title
    self.updateHeader()

    # Init color node combo box <=> make 'Generic Colors' labelmap visible
    model = self.get('LabelmapColorNodeComboBox').sortFilterProxyModel()
    model.setProperty('visibleNodeIDs', slicer.mrmlScene.GetFirstNodeByName('GenericAnatomyColors').GetID())

    # Init transform node
    self.TransformNode = slicer.mrmlScene.GetFirstNodeByName('WorflowTransformNode')
    if self.TransformNode == None:
      self.TransformNode = slicer.vtkMRMLLinearTransformNode()
      self.TransformNode.SetName('WorflowTransformNode')
      self.TransformNode.HideFromEditorsOn()

      transform = vtk.vtkMatrix4x4()
      transform.DeepCopy((-1.0, 0.0, 0.0, 0.0,
                           0.0, -1.0, 0.0, 0.0,
                           0.0, 0.0, 1.0, 0.0,
                           0.0, 0.0, 0.0, 1.0))
      self.TransformNode.ApplyTransformMatrix(transform)
      slicer.mrmlScene.AddNode(self.TransformNode)

    # Workflow page
    self.setupSimpleWorkflow(self.get('WelcomeSimpleWorkflowCheckBox').isChecked())

  # Worflow
  def updateHeader(self):
    # title
    title = self.WorkflowWidget.currentWidget().accessibleName
    self.TitleLabel.setText('<h2>%i: %s</h2>' % (self.WorkflowWidget.currentIndex + 1, title))

    # help
    self.get('HelpCollapsibleButton').setText('%s Help' % title)
    self.get('HelpLabel').setText(self.WorkflowWidget.currentWidget().accessibleDescription)

    # previous
    if self.WorkflowWidget.currentIndex > 0:
      self.get('PreviousPageToolButton').setVisible(True)
      previousIndex = self.WorkflowWidget.currentIndex - 1
      previousWidget = self.WorkflowWidget.widget(previousIndex)

      previous = previousWidget.accessibleName
      self.get('PreviousPageToolButton').setText('< %i: %s' %(previousIndex + 1, previous))
    else:
      self.get('PreviousPageToolButton').setVisible(False)

    # next
    if self.WorkflowWidget.currentIndex < self.WorkflowWidget.count - 1:
      self.get('NextPageToolButton').setVisible(True)
      nextIndex = self.WorkflowWidget.currentIndex + 1
      nextWidget = self.WorkflowWidget.widget(nextIndex)

      next = nextWidget.accessibleName
      self.get('NextPageToolButton').setText('%i: %s >' %(nextIndex + 1, next))
    else:
      self.get('NextPageToolButton').setVisible(False)

  def goToPrevious(self):
    self.WorkflowWidget.setCurrentIndex(self.WorkflowWidget.currentIndex - 1)
    self.updateHeader()

  def goToNext(self):
    self.WorkflowWidget.setCurrentIndex(self.WorkflowWidget.currentIndex + 1)
    self.updateHeader()

  # 0) Welcome
  # Helper function for setting the visibility of a list of widgets
  def setWidgetsVisibility(self, widgets, visible):
    for widget in widgets:
      self.get(widget).setVisible(visible)

  def setupSimpleWorkflow(self, advanced):
    # 1) LabelMap
    # a)
    self.get('LabelmapGoToModulePushButton').setVisible(advanced)

    # b) Merge Labels
    # Hide all but the output
    advancedMergeWidgets = ['MergeLabelsInputLabel', 'MergeLabelsInputNodeComboBox',
                            'BoneLabelsLabel', 'BoneLabelsLineEdit',
                            'BoneLabelLabel', 'BoneLabelComboBox',
                            'SkinLabelsLabel', 'SkinLabelsLineEdit',
                            'SkinLabelLabel', 'SkinLabelComboBox',
                            'MergeLabelsGoToModulePushButton']
    self.setWidgetsVisibility(advancedMergeWidgets, advanced)

    # 2) Model Maker Page
    # a) bone model maker
    # Hide all but the output and the toggle button
    advancedBoneModelMakerWidgets = ['BoneModelMakerInputLabel', 'BoneModelMakerInputNodeComboBox',
                                     'BoneModelMakerLabelsLabel', 'BoneModelMakerLabelsLineEdit',
                                     'BoneModelMakerGoToModulePushButton']
    self.setWidgetsVisibility(advancedBoneModelMakerWidgets, advanced)

    # b) Skin model maker
    # Hide all but the output and the toggle button
    advancedSkinModelMakerWidgets = ['SkinModelMakerNodeInputLabel', 'SkinModelMakerInputNodeComboBox',
                                     'SkinModelMakerThresholdLabel', 'SkinModelMakerThresholdSpinBox',
                                     'SkinModelMakerGoToModulePushButton']
    self.setWidgetsVisibility(advancedSkinModelMakerWidgets, advanced)

    # c) Volume render
    # Just hide it completly
    self.get('VolumeRenderCollapsibleGroupBox').setVisible(advanced)

    # 3) Armature
    # Nothing

    # 4) Weights
    # a) Segment bones
    # Just hide Go To
    self.get('SegmentBonesGoToPushButton').setVisible(advanced)
    # b) Weights
    # Leave only weight folder
    advancedComputeWeightWidgets = ['ArmatureWeightInputVolumeLabel', 'ArmatureWeightInputVolumeNodeComboBox',
                                   'ArmatureWeightArmatureLabel', 'ArmatureWeightAmartureNodeComboBox',
                                   'ArmatureWeightBodyPartitionLabel', 'ArmatureWeightBodyPartitionVolumeNodeComboBox',
                                   'ArmatureWeightGoToPushButton']
    self.setWidgetsVisibility(advancedComputeWeightWidgets, advanced)

    # 5) Pose Page
    # a) Armatures
    # Nothing
    # b) Pose Body
    # hide completely
    advancedPoseBodyWidgets = ['PoseBodyArmaturesLabel', 'PoseBodyArmatureInputNodeComboBox',
                               'PoseBodySurfaceInputLabel', 'PoseBodySurfaceInputComboBox',
                               'PoseBodyWeightInputFolderLabel', 'PoseBodyWeightInputDirectoryButton',
                               'PoseBodyGoToPushButton']
    self.setWidgetsVisibility(advancedPoseBodyWidgets, advanced)

    # 6) Resample
    # TO DO !!!

  # 1) Bone Segmentation
  #     a) Labelmap
  def updateLabelmap(self, node, event):
    volumeNode = self.get('LabelmapVolumeNodeComboBox').currentNode()
    self.setupMergeLabels(volumeNode)

  def setupLabelmap(self, volumeNode):
    if volumeNode == None:
      return

    if self.oldLabelMapVolumeNode != None and self.oldLabelMapVolumeNode != volumeNode:
      self.oldLabelMapVolumeNode.RemoveObserver(self.volumeNodeLabelMapTag)
      self.oldLabelMapVolumeNode.GetDisplayNode().RemoveObserver(self.labelmapDisplayNodeLabelMapTag)

    self.volumeNodeLabelMapTag = volumeNode.AddObserver('ModifiedEvent', self.updateLabelmap)
    self.labelmapDisplayNodeLabelMapTag = volumeNode.GetDisplayNode().AddObserver('ModifiedEvent', self.updateLabelmap)
    self.oldLabelMapVolumeNode = volumeNode

  def applyColorNode(self):
    volumeNode = self.get('LabelmapVolumeNodeComboBox').currentNode()
    if volumeNode == None:
      return

    self.get('LabelMapApplyColorNodePushButton').setChecked(True)
      
    colorNode = self.get('LabelmapColorNodeComboBox').currentNode()
    volumesLogic = slicer.modules.volumes.logic()

    wasModifying = volumeNode.StartModify()
    volumesLogic.SetVolumeAsLabelMap(volumeNode, colorNode != None) # Greyscale is None

    labelmapDisplayNode = volumeNode.GetDisplayNode()
    if colorNode != None:
      labelmapDisplayNode.SetAndObserveColorNodeID(colorNode.GetID())
    volumeNode.EndModify(wasModifying)

    self.setupMergeLabels(volumeNode)
    self.get('LabelMapApplyColorNodePushButton').setChecked(False)

  def openLabelmapModule(self):
    self.openModule('Volumes')

  #    b) Transform
  def runTransform(self):
    volumeNode = self.get('LabelmapVolumeNodeComboBox').currentNode()
    if volumeNode == None:
      return

    self.get('TransformApplyPushButton').setChecked(True)
      
    volumeNode.SetAndObserveTransformNodeID(self.TransformNode.GetID())
    transformLogic = slicer.modules.transforms.logic()

    if transformLogic.hardenTransform(volumeNode):
      print "Transform succesful !"
    else:
      print "Transform failure !"
      
    self.get('TransformApplyPushButton').setChecked(False)

  #    c) Merge Labels
  def updateMergeLabels(self, node, event):
    volumeNode = self.get('MergeLabelsInputNodeComboBox').currentNode()
    if node.IsA('vtkMRMLScalarVolumeNode') and node != volumeNode:
      return
    elif node.IsA('vtkMRMLVolumeDisplayNode'):
      if node != volumeNode.GetDisplayNode():
        return
    self.setupMergeLabels(volumeNode)

  def setupMergeLabels(self, volumeNode):
    if volumeNode == None:
      return
    labelmapDisplayNode = volumeNode.GetDisplayNode()
    colorNode = labelmapDisplayNode.GetColorNode()
    if colorNode == None:
      self.get('BoneLabelComboBox').setMRMLColorNode(None)
      self.get('SkinLabelComboBox').setMRMLColorNode(None)
      self.get('BoneLabelsLineEdit').setText('')
      self.get('BoneLabelComboBox').setCurrentColor(None)
      self.get('SkinLabelsLineEdit').setText('')
      self.get('SkinLabelComboBox').setCurrentColor(None)

      volumeNode.RemoveObserver(self.volumeNodeMergeLabelsTag)
      labelmapDisplayNode.RemoveObserver(self.labelmapDisplayNodeMergeLabelsTag)
    else:
      self.get('BoneLabelComboBox').setMRMLColorNode(colorNode)
      self.get('SkinLabelComboBox').setMRMLColorNode(colorNode)
      boneLabels = self.searchLabels(colorNode, 'bone')
      boneLabels.update(self.searchLabels(colorNode, 'vertebr'))
      self.get('BoneLabelsLineEdit').setText(', '.join(str( val ) for val in boneLabels.keys()))
      boneLabel = self.bestLabel(boneLabels, 'bone')
      self.get('BoneLabelComboBox').setCurrentColor(boneLabel)
      skinLabels = self.searchLabels(colorNode, 'skin')
      self.get('SkinLabelsLineEdit').setText(', '.join(str(val) for val in skinLabels.keys()))
      skinLabel = self.bestLabel(skinLabels, 'skin')
      self.get('SkinLabelComboBox').setCurrentColor(skinLabel)

      self.volumeNodeMergeLabelsTag = volumeNode.AddObserver('ModifiedEvent', self.updateMergeLabels)
      self.labelmapDisplayNodeMergeLabelsTag = labelmapDisplayNode.AddObserver('ModifiedEvent', self.updateMergeLabels)

  def searchLabels(self, colorNode, label):
    """ Search the color node for all the labels that contain the word 'label'
    """
    labels = {}
    for index in range(colorNode.GetNumberOfColors()):
      if label in colorNode.GetColorName(index).lower():
        labels[index] = colorNode.GetColorName(index)
    return labels

  def bestLabel(self, labels, label):
    """ Return the label from a [index, colorName] map that fits the best the
         label name
    """
    if (len(labels) == 0):
      return -1

    for key in labels.keys():
      if labels[key].lower().startswith(label):
        return key
    return labels.keys()[0]

  def runMergeLabels(self):
    boneLabels = self.get('BoneLabelsLineEdit').text
    skinLabels = self.get('SkinLabelsLineEdit').text
    parameters = {}
    parameters["InputVolume"] = self.get('MergeLabelsInputNodeComboBox').currentNode()
    parameters["OutputVolume"] = self.get('MergeLabelsOutputNodeComboBox').currentNode()
    # That's my dream:
    #parameters["InputLabelNumber"] = len(boneLabels.split(','))
    #parameters["InputLabelNumber"] = len(skinLabels.split(','))
    #parameters["InputLabel"] = boneLabels
    #parameters["InputLabel"] = skinLabels
    #parameters["OutputLabel"] = self.get('BoneLabelComboBox').currentColor
    #parameters["OutputLabel"] = self.get('SkinLabelComboBox').currentColor
    # But that's how it is done for now
    parameters["InputLabelNumber"] = str(len(boneLabels.split(','))) + ', ' + str(len(skinLabels.split(',')))
    parameters["InputLabel"] = boneLabels + ', ' + skinLabels
    parameters["OutputLabel"] = str(self.get('BoneLabelComboBox').currentColor) + ', ' + str(self.get('SkinLabelComboBox').currentColor)
    cliNode = None

    self.get('MergeLabelsApplyPushButton').setChecked(True)

    cliNode = slicer.cli.run(slicer.modules.changelabel, cliNode, parameters, wait_for_completion = True)
    status = cliNode.GetStatusString()
    if status == 'Completed':
      print 'MergeLabels completed'

      # apply label map
      newNode = self.get('MergeLabelsOutputNodeComboBox').currentNode()
      colorNode = self.get('LabelmapColorNodeComboBox').currentNode()
      if newNode != None and colorNode != None:
        volumesLogic = slicer.modules.volumes.logic()
        wasModifying = newNode.StartModify()
        volumesLogic.SetVolumeAsLabelMap(newNode, True)

        newNode.GetDisplayNode().SetAndObserveColorNodeID(colorNode.GetID())
        newNode.EndModify(wasModifying)

    else:
      print 'MergeLabels failed'

    self.get('MergeLabelsApplyPushButton').setChecked(False)

  def openMergeLabelsModule(self):
    self.openModule('ChangeLabel')

  # 2) Model Maker
  #     a) Bone Model Maker
  def setupBoneModelMakerLabels(self):
    """ Update the labels of the bone model maker
    """
    labels = []
    labels.append(self.get('BoneLabelComboBox').currentColor)
    self.get('BoneModelMakerLabelsLineEdit').setText(', '.join(str(val) for val in labels))

  def runBoneModelMaker(self):
    parameters = {}
    parameters["InputVolume"] = self.get('BoneModelMakerInputNodeComboBox').currentNode()
    parameters["ModelSceneFile"] = self.get('BoneModelMakerOutputNodeComboBox').currentNode()
    parameters["Labels"] = self.get('BoneModelMakerLabelsLineEdit').text
    parameters["Name"] = 'Skeleton'
    parameters['GenerateAll'] = False
    parameters["JointSmoothing"] = False
    parameters["SplitNormals"] = True
    parameters["PointNormals"] = True
    parameters["SkipUnNamed"] = True
    parameters["Decimate"] = 0.25
    parameters["Smooth"] = 10
    cliNode = None

    self.get('BoneModelMakerApplyPushButton').setChecked(True)

    cliNode = slicer.cli.run(slicer.modules.modelmaker, cliNode, parameters, wait_for_completion = True)
    status = cliNode.GetStatusString()
    if status == 'Completed':
      print 'ModelMaker completed'

    else:
      print 'ModelMaker failed'

    self.get('BoneModelMakerApplyPushButton').setChecked(False)

  def openBoneModelMakerModule(self):
    self.openModule('ModelMaker')

  #     b) Skin Model Maker
  def setupSkinModelMakerLabels(self, volumeNode):
    """ Update the labels of the skin model maker
    """
    if volumeNode == None:
      return

    labelmapDisplayNode = volumeNode.GetDisplayNode()
    if labelmapDisplayNode == None:
      return

    colorNode = labelmapDisplayNode.GetColorNode()
    if colorNode == None:
      self.get('SkinModelMakerLabelsLineEdit').setText('')
    else:
      airLabels = self.searchLabels(colorNode, 'air')
      if len(airLabels) > 0:
        self.get('SkinModelMakerThresholdSpinBox').setValue( min(airLabels) + 0.1 ) # highly probable outside is lowest label
      else:
        self.get('SkinModelMakerThresholdSpinBox').setValue(0.1) # highly probable outside is 0

  def runSkinModelMaker(self):
    parameters = {}
    parameters["InputVolume"] = self.get('SkinModelMakerInputNodeComboBox').currentNode()
    parameters["OutputGeometry"] = self.get('SkinModelMakerOutputNodeComboBox').currentNode()
    parameters["Threshold"] = self.get('SkinModelMakerThresholdSpinBox').value + 0.1
    #parameters["SplitNormals"] = True
    #parameters["PointNormals"] = True
    #parameters["Decimate"] = 0.25
    parameters["Smooth"] = 10
    cliNode = None

    self.get('SkinModelMakerApplyPushButton').setChecked(True)

    cliNode = slicer.cli.run(slicer.modules.grayscalemodelmaker, cliNode, parameters, wait_for_completion = True)
    status = cliNode.GetStatusString()
    if status == 'Completed':
      print 'Grayscale ModelMaker completed'

      self.get('SkinModelMakerOutputNodeComboBox').currentNode().GetModelDisplayNode().SetOpacity(0.2)

    else:
      print 'Grayscale ModelMaker failed'

    self.get('SkinModelMakerApplyPushButton').setChecked(False)

  def openSkinModelMakerModule(self):
    self.openModule('GrayscaleModelMaker')

  def updateSkinNodeVisibility(self):
    skinModel = self.get('SkinModelMakerOutputNodeComboBox').currentNode()
    if skinModel != None:
      skinModel.SetDisplayVisibility(not skinModel.GetDisplayVisibility())

  #     c) Volume Render
  def updateVolumeRender(self, volumeNode, event):
    if volumeNode != self.get('VolumeRenderInputNodeComboBox').currentNode():
      return
    self.setupVolumeRender(volumeNode)

  def setupVolumeRender(self, volumeNode):
    if volumeNode == None:
      return
    displayNode = volumeNode.GetNthDisplayNodeByClass(0, 'vtkMRMLVolumeRenderingDisplayNode')
    visible = False
    if displayNode != None:
      visible = displayNode.GetVisibility()
    self.get('VolumeRenderCheckBox').setChecked(visible)
    self.setupVolumeRenderLabels()
    volumeNode.AddObserver('ModifiedEvent', self.updateVolumeRender)

  def setupVolumeRenderLabels(self):
    """ Update the labels of the volume rendering
    """
    labels = []
    labels.append(self.get('BoneLabelComboBox').currentColor)
    labels.append(self.get('SkinLabelComboBox').currentColor)
    self.get('VolumeRenderLabelsLineEdit').setText(', '.join(str(val) for val in labels))

  def getVolumeRenderLabels(self):
    return self.get('VolumeRenderLabelsLineEdit').text.split(', ')

  def updateVolumeRenderLabels(self):
    """ Update the LUT used to volume render the labelmap
    """
    if not self.get('VolumeRenderCheckBox').isChecked():
      return
    volumeNode = self.get('VolumeRenderInputNodeComboBox').currentNode()
    displayNode = volumeNode.GetNthDisplayNodeByClass(0, 'vtkMRMLVolumeRenderingDisplayNode')
    volumePropertyNode = displayNode.GetVolumePropertyNode()
    opacities = volumePropertyNode.GetScalarOpacity()
    labels = self.getVolumeRenderLabels()
    for i in range(opacities.GetSize()):
      node = [0, 0, 0, 0]
      opacities.GetNodeValue(i, node)
      if str(i) in labels:
        node[1] = 0.5
        node[3] = 1
      else:
        node[1] = 0
        node[3] = 1
      opacities.SetNodeValue(i, node)
    opacities.Modified()

  def runVolumeRender(self, show):
    volumeNode = self.get('VolumeRenderInputNodeComboBox').currentNode()
    displayNode = volumeNode.GetNthDisplayNodeByClass(0, 'vtkMRMLVolumeRenderingDisplayNode')
    if not show:
      if displayNode == None:
        return
      displayNode.SetVisibility(0)
    else:
      volumeRenderingLogic = slicer.modules.volumerendering.logic()
      if displayNode == None:
        displayNode = volumeRenderingLogic.CreateVolumeRenderingDisplayNode()
        slicer.mrmlScene.AddNode(displayNode)
        displayNode.UnRegister(volumeRenderingLogic)
        volumeRenderingLogic.UpdateDisplayNodeFromVolumeNode(displayNode, volumeNode)
        volumeNode.AddAndObserveDisplayNodeID(displayNode.GetID())
      else:
        volumeRenderingLogic.UpdateDisplayNodeFromVolumeNode(displayNode, volumeNode)
      self.updateVolumeRenderLabels()
      volumePropertyNode = displayNode.GetVolumePropertyNode()
      volumeProperty = volumePropertyNode.GetVolumeProperty()
      volumeProperty.SetShade(0)
      displayNode.SetVisibility(1)

  def openVolumeRenderModule(self):
    self.openModule('VolumeRendering')

  # 3) Armatures first part
  def openArmaturesModule(self):
    # First reset focal view around volumes
    manager = slicer.app.layoutManager()
    for i in range(0, manager.threeDViewCount):
      manager.threeDWidget(i).threeDView().resetFocalPoint()

    # Switch to 3D View only
    manager.setLayout(slicer.vtkMRMLLayoutNode().SlicerLayoutOneUp3DView)

    # Finaly open armature module
    self.openModule('Armatures')

  # 4) Armatures Weight and Bones
  #  a) Armatures Bones
  def runSegmentBones(self):
    parameters = {}
    parameters["RestLabelmap"] = self.get('SegmentBonesInputVolumeNodeComboBox').currentNode()
    parameters["ArmaturePoly"] = self.get('SegmentBonesAmartureNodeComboBox').currentNode()
    parameters["BodyPartition"] = self.get('SegmentBonesOutputVolumeNodeComboBox').currentNode()
    #parameters["Padding"] = 1
    #parameters["Debug"] = False
    #parameters["ArmatureInRAS"] = False
    cliNode = None

    self.get('SegmentBonesApplyPushButton').setChecked(True)

    cliNode = slicer.cli.run(slicer.modules.armaturebones, cliNode, parameters, wait_for_completion = True)
    status = cliNode.GetStatusString()
    if status == 'Completed':
      print 'Armature Bones completed'
    else:
      print 'Armature Bones failed'

    self.get('SegmentBonesApplyPushButton').setChecked(False)

  def openSegmentBonesModule(self):
    self.openModule('ArmatureWeight')

    #  b) Armature Weight
  def runArmatureWeight(self):
    parameters = {}
    parameters["RestLabelmap"] = self.get('ArmatureWeightInputVolumeNodeComboBox').currentNode()
    parameters["ArmaturePoly"] = self.get('ArmatureWeightAmartureNodeComboBox').currentNode()
    parameters["BodyPartition"] = self.get('ArmatureWeightBodyPartitionVolumeNodeComboBox').currentNode()
    parameters["WeightDirectory"] = str(self.get('ArmatureWeightOutputDirectoryButton').directory)
    #parameters["FirstEdge"] = 0
    #parameters["LastEdge"] = -1
    #parameters["BinaryWeight"] = False
    #parameters["SmoothingIteration"] = 10
    #parameters["Debug"] = False
    #parameters["RunSequential"] = False
    cliNode = None

    self.get('ArmatureWeightApplyPushButton').setChecked(True)

    cliNode = slicer.cli.run(slicer.modules.armatureweight, cliNode, parameters, wait_for_completion = True)
    status = cliNode.GetStatusString()
    if status == 'Completed':
      print 'Armature Weight completed'
    else:
      print 'Armature Weight failed'

    self.get('ArmatureWeightApplyPushButton').setChecked(False)

  def openArmatureWeightModule(self):
    self.openModule('ArmatureWeight')

  # 5) (Pose) Armature And Pose Body
  # a) (Pose) Armatures
  def openPosedArmatureModule(self):
    self.openModule('Armatures')

  # b) Pose Body
  def updatePoseBodyFromCLI(self, node, event):
    if node != self.PoseBodyCLI:
      return

    self.get('PoseBodyArmatureInputNodeComboBox').setCurrentNode(
      slicer.mrmlScene.GetNodeByID(
        self.PoseBodyCLI.GetParameterAsString('ArmaturePoly')))
    self.get('PoseBodySurfaceInputComboBox').setCurrentNode(
      slicer.mrmlScene.GetNodeByID(
        self.PoseBodyCLI.GetParameterAsString('SurfaceInput')))
    self.get('PoseBodyWeightInputDirectoryButton').directory = (
      self.PoseBodyCLI.GetParameterAsString('WeightDirectory'))
    self.get('PoseBodySurfaceOutputNodeComboBox').setCurrentNode(
      slicer.mrmlScene.GetNodeByID(
        self.PoseBodyCLI.GetParameterAsString('OutputSurface')))

  def setupPoseBody(self):
    # Setup CLI node on input changed or apply changed
    self.PoseBodyCLI.SetAutoRunMode(self.PoseBodyCLI.AutoRunOnAnyInputEvent)

    parametersAreValid = (self.get('PoseBodyArmatureInputNodeComboBox').currentNode() != None
                          and self.get('PoseBodySurfaceInputComboBox').currentNode() != None
                          and self.get('PoseBodySurfaceOutputNodeComboBox').currentNode() != None)
    parameters = {}
    if parametersAreValid == True:
      parameters["ArmaturePoly"] = self.get('PoseBodyArmatureInputNodeComboBox').currentNode()
      parameters["SurfaceInput"] = self.get('PoseBodySurfaceInputComboBox').currentNode()
      parameters["WeightDirectory"] = str(self.get('PoseBodyWeightInputDirectoryButton').directory)
      parameters["OutputSurface"] = self.get('PoseBodySurfaceOutputNodeComboBox').currentNode()
      parameters["IsSurfaceInRAS"] = False
      parameters["IsArmatureInRAS"] = False
      parameters["LinearBlend"] = True

    slicer.cli.setNodeParameters(self.PoseBodyCLI, parameters)

  def runPoseBody(self):
    if self.get('PoseBodyApplyPushButton').checkState == False:
      if self.PoseBodyCLI.IsBusy() == False:
        self.get('PoseBodyApplyPushButton').setChecked(True)
        slicer.modules.posebody.logic().ApplyAndWait(self.PoseBodyCLI)
        self.get('PoseBodyApplyPushButton').setChecked(False)
      else:
        self.PoseBodyCLI.Cancel()
    else:
      self.PoseBodyCLI.SetAutoRun(self.get('PoseBodyApplyPushButton').isChecked())

  def openPoseBodyModule(self):
    self.openModule('PoseBody')

  def setWeightDirectory(self, dir):
    if self.get('PoseBodyWeightInputDirectoryButton').directory != dir:
      self.get('PoseBodyWeightInputDirectoryButton').directory = dir

  # 6) Resample NOTE: SHOULD BE LAST STEP
  def runResample(self):
    print('Resample')

  # =================== END ==============
  def get(self, objectName):
    return self.findWidget(self.widget, objectName)

  def findWidget(self, widget, objectName):
    if widget.objectName == objectName:
        return widget
    else:
        children = []
        for w in widget.children():
            resulting_widget = self.findWidget(w, objectName)
            if resulting_widget:
                return resulting_widget
        return None

  def openModule(self, moduleName):
    slicer.util.selectModule(moduleName)

  def reloadModule(self,moduleName="Workflow"):
    """Generic reload method for any scripted module.
    ModuleWizard will subsitute correct default moduleName.
    """
    import imp, sys, os, slicer

    widgetName = moduleName + "Widget"

    # reload the source code
    # - set source file path
    # - load the module to the global space
    filePath = eval('slicer.modules.%s.path' % moduleName.lower())
    p = os.path.dirname(filePath)
    if not sys.path.__contains__(p):
      sys.path.insert(0,p)
    fp = open(filePath, "r")
    globals()[moduleName] = imp.load_module(
        moduleName, fp, filePath, ('.py', 'r', imp.PY_SOURCE))
    fp.close()

    # rebuild the widget
    # - find and hide the existing widget
    # - create a new widget in the existing parent
    parent = slicer.util.findChildren(name='%s Reload' % moduleName)[0].parent()
    for child in parent.children():
      try:
        child.hide()
      except AttributeError:
        pass

    self.layout.removeWidget(self.widget)
    self.widget.deleteLater()
    self.widget = None

    # Remove spacer items
    item = parent.layout().itemAt(0)
    while item:
      parent.layout().removeItem(item)
      item = parent.layout().itemAt(0)
    # create new widget inside existing parent
    globals()[widgetName.lower()] = eval(
        'globals()["%s"].%s(parent)' % (moduleName, widgetName))
    globals()[widgetName.lower()].setup()

  # =================== END ==============

class WorkflowLogic:
  """Implement the logic to calculate label statistics.
  Nodes are passed in as arguments.
  Results are stored as 'statistics' instance variable.
  """
  def __init__(self):
    return

class Slicelet(object):
  """A slicer slicelet is a module widget that comes up in stand alone mode
  implemented as a python class.
  This class provides common wrapper functionality used by all slicer modlets.
  """
  # TODO: put this in a SliceletLib
  # TODO: parse command line arge


  def __init__(self, widgetClass=None):
    self.parent = qt.QFrame()
    self.parent.setLayout( qt.QVBoxLayout() )

    # TODO: should have way to pop up python interactor
    self.buttons = qt.QFrame()
    self.buttons.setLayout( qt.QHBoxLayout() )
    self.parent.layout().addWidget(self.buttons)
    self.addDataButton = qt.QPushButton("Add Data")
    self.buttons.layout().addWidget(self.addDataButton)
    self.addDataButton.connect("clicked()",slicer.app.ioManager().openAddDataDialog)
    self.loadSceneButton = qt.QPushButton("Load Scene")
    self.buttons.layout().addWidget(self.loadSceneButton)
    self.loadSceneButton.connect("clicked()",slicer.app.ioManager().openLoadSceneDialog)

    if widgetClass:
      self.widget = widgetClass(self.parent)
      self.widget.setup()
    self.parent.show()

class WorkflowSlicelet(Slicelet):
  """ Creates the interface when module is run as a stand alone gui app.
  """

  def __init__(self):
    super(WorkflowSlicelet,self).__init__(WorkflowWidget)


if __name__ == "__main__":
  # TODO: need a way to access and parse command line arguments
  # TODO: ideally command line args should handle --xml

  import sys
  print( sys.argv )

  slicelet = WorkflowSlicelet()
