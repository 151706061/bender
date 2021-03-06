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

    self.Observations = []

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

    # Transform variables
    self.TransformNode = None

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
    self.get('BoneModelMakerGoToModelsModulePushButton').connect('clicked()', self.openModelsModule)
    self.get('BoneModelMakerGoToModulePushButton').connect('clicked()', self.openBoneModelMakerModule)
    # b) Skin Model Maker
    self.get('SkinModelMakerInputNodeComboBox').connect('currentNodeChanged(vtkMRMLNode*)', self.setupSkinModelMakerLabels)
    self.get('SkinModelMakerToggleVisiblePushButtton').connect('clicked()', self.updateSkinNodeVisibility)
    self.get('SkinModelMakerApplyPushButton').connect('clicked()', self.runSkinModelMaker)
    self.get('SkinModelMakerGoToModelsModulePushButton').connect('clicked()', self.openModelsModule)
    self.get('SkinModelMakerGoToModulePushButton').connect('clicked()', self.openSkinModelMakerModule)
    # c) Volume Render
    self.get('BoneLabelComboBox').connect('currentColorChanged(int)', self.setupVolumeRenderLabels)
    self.get('SkinLabelComboBox').connect('currentColorChanged(int)', self.setupVolumeRenderLabels)
    self.get('VolumeRenderInputNodeComboBox').connect('currentNodeChanged(vtkMRMLNode*)', self.setupVolumeRender)
    self.get('VolumeRenderLabelsLineEdit').connect('editingFinished()', self.updateVolumeRenderLabels)
    self.get('VolumeRenderCheckBox').connect('toggled(bool)',self.runVolumeRender)
    self.get('VolumeRenderGoToModulePushButton').connect('clicked()', self.openVolumeRenderModule)
    # 3) Armatures
    self.get('ArmaturesToggleVisiblePushButtton').connect('clicked()', self.updateSkinNodeVisibility)
    self.get('ArmaturesLoadArmaturePushButton').connect('clicked()', self.loadArmatureFile)
    self.get('ArmaturesGoToPushButton').connect('clicked()', self.openArmaturesModule)
    # 4) Armature Weight and Bones
    # a) Armatures Bones
    self.get('SegmentBonesApplyPushButton').connect('clicked()',self.runSegmentBones)
    self.get('SegmentBonesGoToPushButton').connect('clicked()', self.openSegmentBonesModule)
    # b) Armatures Weight
    self.get('ArmatureWeightApplyPushButton').connect('clicked()',self.runArmatureWeight)
    self.get('ArmatureWeightGoToPushButton').connect('clicked()', self.openArmatureWeightModule)
    # 5) (Pose) Armature And Pose Body
    # a) Eval Weight
    self.get('EvalWeightApplyPushButton').connect('clicked()', self.runEvalWeight)
    self.get('EvalWeightGoToPushButton').connect('clicked()', self.openEvalWeight)
    self.get('EvalWeightWeightDirectoryButton').connect('directoryChanged(QString)', self.setWeightDirectory)
    # b) (Pose) Armatures
    self.get('PoseArmaturesGoToPushButton').connect('clicked()', self.openPosedArmatureModule)
    # c) Pose Body
    self.get('PoseBodySurfaceOutputNodeComboBox').connect('currentNodeChanged(vtkMRMLNode*)', self.poseBodyParameterChanged)
    self.get('PoseBodyWeightInputDirectoryButton').connect('directoryChanged(QString)', self.poseBodyParameterChanged)
    self.get('PoseBodySurfaceInputComboBox').connect('currentNodeChanged(vtkMRMLNode*)', self.poseBodyParameterChanged)
    self.get('PoseBodyArmatureInputNodeComboBox').connect('currentNodeChanged(vtkMRMLNode*)', self.poseBodyParameterChanged)
    self.get('PoseBodyApplyPushButton').connect('clicked()', self.runPoseBody)

    self.get('PoseBodyGoToPushButton').connect('clicked()', self.openPoseBodyModule)
    self.get('ArmatureWeightOutputDirectoryButton').connect('directoryChanged(QString)', self.setWeightDirectory)

    self.get('PoseBodySurfaceInputComboBox').connect('currentNodeChanged(vtkMRMLNode*)', self.createOutputSurface)
    # 6) Resample
    self.get('PoseLabelmapApplyPushButton').connect('clicked()', self.runPoseLabelmap)
    self.get('PoseLabelmapGoToPushButton').connect('clicked()', self.openPoseLabelmap)

    self.openPage = { 0 : self.openAdjustPage,
                      1 : self.openExtractPage,
                      2 : self.openCreateArmaturePage,
                      3 : self.openSkinningPage,
                      4 : self.openPoseArmaturePage,
                      5 : self.openPoseLabelmapPage
                      }
    # --------------------------------------------------------------------------
    # Initialize all the MRML aware GUI elements.
    # Lots of setup methods are called from this line
    self.widget.setMRMLScene(slicer.mrmlScene)

    # Init title
    self.updateHeader()

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
    self.get('AdvancedPropertiesWidget').setVisible(self.get('ExpandAdvancedPropertiesButton').isChecked())

  # Worflow
  def updateHeader(self):
    # title
    title = self.WorkflowWidget.currentWidget().accessibleName
    self.TitleLabel.setText('<h2>%i) %s</h2>' % (self.WorkflowWidget.currentIndex + 1, title))

    # help
    self.get('HelpCollapsibleButton').setText('Help')
    self.get('HelpLabel').setText(self.WorkflowWidget.currentWidget().accessibleDescription)

    # previous
    if self.WorkflowWidget.currentIndex > 0:
      self.get('PreviousPageToolButton').setVisible(True)
      previousIndex = self.WorkflowWidget.currentIndex - 1
      previousWidget = self.WorkflowWidget.widget(previousIndex)

      previous = previousWidget.accessibleName
      self.get('PreviousPageToolButton').setText('< %i) %s' %(previousIndex + 1, previous))
    else:
      self.get('PreviousPageToolButton').setVisible(False)

    # next
    if self.WorkflowWidget.currentIndex < self.WorkflowWidget.count - 1:
      self.get('NextPageToolButton').setVisible(True)
      nextIndex = self.WorkflowWidget.currentIndex + 1
      nextWidget = self.WorkflowWidget.widget(nextIndex)

      next = nextWidget.accessibleName
      self.get('NextPageToolButton').setText('%i) %s >' %(nextIndex + 1, next))
    else:
      self.get('NextPageToolButton').setVisible(False)
    self.openPage[self.WorkflowWidget.currentIndex]()

  def goToPrevious(self):
    self.WorkflowWidget.setCurrentIndex(self.WorkflowWidget.currentIndex - 1)
    self.updateHeader()

  def goToNext(self):
    self.WorkflowWidget.setCurrentIndex(self.WorkflowWidget.currentIndex + 1)
    self.updateHeader()

  # 0) Welcome
  def openWelcomePage(self):
    print('welcome')

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
                                     'BoneModelMakerGoToModelsModulePushButton',
                                     'BoneModelMakerGoToModulePushButton']
    self.setWidgetsVisibility(advancedBoneModelMakerWidgets, advanced)

    # b) Skin model maker
    # Hide all but the output and the toggle button
    advancedSkinModelMakerWidgets = ['SkinModelMakerNodeInputLabel', 'SkinModelMakerInputNodeComboBox',
                                     'SkinModelMakerThresholdLabel', 'SkinModelMakerThresholdSpinBox',
                                     'SkinModelMakerGoToModelsModulePushButton',
                                     'SkinModelMakerGoToModulePushButton']
    self.setWidgetsVisibility(advancedSkinModelMakerWidgets, advanced)

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
    # b) Eval Weight
    advancedEvalWeightWidgets = ['EvalWeightInputSurfaceLabel', 'EvalWeightInputNodeComboBox',
                                 'EvalWeightWeightDirectoryLabel', 'EvalWeightWeightDirectoryButton',
                                 'EvalWeightGoToPushButton']
    self.setWidgetsVisibility(advancedEvalWeightWidgets, advanced)

    # c) Pose Body
    # hide almost completely
    advancedPoseBodyWidgets = ['PoseBodyArmaturesLabel', 'PoseBodyArmatureInputNodeComboBox',
                               'PoseBodySurfaceInputLabel', 'PoseBodySurfaceInputComboBox',
                               'PoseBodyWeightInputFolderLabel', 'PoseBodyWeightInputDirectoryButton',
                               'PoseBodyGoToPushButton']
    self.setWidgetsVisibility(advancedPoseBodyWidgets, advanced)

    # 6) Resample
    # Hide all but output
    advancedPoseLabemapWidgets = ['PoseLabelmapInputLabel', 'PoseLabelmapInputNodeComboBox',
                                  'PoseLabelmapArmatureLabel', 'PoseLabelmapArmatureNodeComboBox',
                                  'PoseLabelmapWeightDirectoryLabel', 'PoseLabelmapWeightDirectoryButton',
                                  'PoseLabelmapGoToPushButton']
    self.setWidgetsVisibility(advancedPoseLabemapWidgets, advanced)

  # 1) Bone Segmentation
  def openAdjustPage(self):
    # Switch to 3D View only
    slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)

  #     a) Labelmap
  def updateLabelmap(self, node, event):
    volumeNode = self.get('LabelmapVolumeNodeComboBox').currentNode()
    if node != volumeNode and node != volumeNode.GetDisplayNode():
      return
    self.setupLabelmap(volumeNode)
    self.setupMergeLabels(volumeNode)

  def setupLabelmap(self, volumeNode):
    if volumeNode == None:
      return

    # Init color node combo box <=> make 'Generic Colors' labelmap visible
    model = self.get('LabelmapColorNodeComboBox').sortFilterProxyModel()
    visibleNodeIDs = []
    visibleNodeIDs.append(slicer.mrmlScene.GetFirstNodeByName('GenericColors').GetID())
    visibleNodeIDs.append(slicer.mrmlScene.GetFirstNodeByName('GenericAnatomyColors').GetID())
    model.visibleNodeIDs = visibleNodeIDs

    # Labelmapcolornode should get its scene before the volume node selector
    # gets it. That way, setCurrentNode can work at first
    self.get('LabelmapColorNodeComboBox').setCurrentNode(volumeNode.GetDisplayNode().GetColorNode())
    self.addObserver(volumeNode, 'ModifiedEvent', self.updateLabelmap)
    self.addObserver(volumeNode.GetDisplayNode(), 'ModifiedEvent', self.updateLabelmap)

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

    self.get('MergeLabelsInputNodeComboBox').setCurrentNode(
      self.get('LabelmapVolumeNodeComboBox').currentNode())
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
    self.get('MergeLabelsInputNodeComboBox').addAttribute('vtkMRMLScalarVolumeNode','LabelMap','1')
    self.get('MergeLabelsOutputNodeComboBox').addAttribute('vtkMRMLScalarVolumeNode','LabelMap','1')
    labelmapDisplayNode = volumeNode.GetDisplayNode()
    self.removeObservers(self.updateMergeLabels)
    colorNode = labelmapDisplayNode.GetColorNode()
    if colorNode == None:
      self.get('BoneLabelComboBox').setMRMLColorNode(None)
      self.get('SkinLabelComboBox').setMRMLColorNode(None)
      self.get('BoneLabelsLineEdit').setText('')
      self.get('BoneLabelComboBox').setCurrentColor(None)
      self.get('SkinLabelsLineEdit').setText('')
      self.get('SkinLabelComboBox').setCurrentColor(None)

    else:
      self.get('BoneLabelComboBox').setMRMLColorNode(colorNode)
      self.get('SkinLabelComboBox').setMRMLColorNode(colorNode)
      boneLabels = self.searchLabels(colorNode, 'bone')
      boneLabels.update(self.searchLabels(colorNode, 'vertebr'))
      self.get('BoneLabelsLineEdit').setText(', '.join(str( val ) for val in boneLabels.keys()))
      boneLabel = self.bestLabel(boneLabels, ['bone', 'cancellous'])
      self.get('BoneLabelComboBox').setCurrentColor(boneLabel)
      skinLabels = self.searchLabels(colorNode, 'skin')
      self.get('SkinLabelsLineEdit').setText(', '.join(str(val) for val in skinLabels.keys()))
      skinLabel = self.bestLabel(skinLabels, ['skin'])
      self.get('SkinLabelComboBox').setCurrentColor(skinLabel)

      self.addObserver(volumeNode, 'ModifiedEvent', self.updateMergeLabels)
      self.addObserver(labelmapDisplayNode, 'ModifiedEvent', self.updateMergeLabels)

  def searchLabels(self, colorNode, label):
    """ Search the color node for all the labels that contain the word 'label'
    """
    labels = {}
    for index in range(colorNode.GetNumberOfColors()):
      if label in colorNode.GetColorName(index).lower():
        labels[index] = colorNode.GetColorName(index)
    return labels

  def bestLabel(self, labels, labelNames):
    """ Return the label from a [index, colorName] map that fits the best the
         label name
    """
    bestLabels = labels
    if (len(bestLabels) == 0):
      return -1

    labelIndex = 0
    for labelName in labelNames:
      newBestLabels = {}
      for key in bestLabels.keys():
        startswith = bestLabels[key].lower().startswith(labelName)
        contains = labelName in bestLabels[key].lower()
        if (labelIndex == 0 and startswith) or (labelIndex > 0 and contains):
          newBestLabels[key] = bestLabels[key]
      if len(newBestLabels) == 1:
        return newBestLabels.keys()[0]
      bestLabels = newBestLabels
      labelIndex = labelIndex + 1
    return bestLabels.keys()[0]

  def mergeLabelsParameters(self):
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
    return parameters

  def runMergeLabels(self):
    cliNode = self.getCLINode(slicer.modules.changelabel)
    parameters = self.mergeLabelsParameters()
    self.get('MergeLabelsApplyPushButton').setChecked(True)
    cliNode = slicer.cli.run(slicer.modules.changelabel, cliNode, parameters, wait_for_completion = True)
    self.get('MergeLabelsApplyPushButton').setChecked(False)

    if cliNode.GetStatusString() == 'Completed':
      print 'MergeLabels completed'
      # apply label map
      newNode = self.get('MergeLabelsOutputNodeComboBox').currentNode()
      colorNode = self.get('LabelmapColorNodeComboBox').currentNode()
      if newNode != None and colorNode != None:
        newNode.GetDisplayNode().SetAndObserveColorNodeID(colorNode.GetID())

    else:
      print 'MergeLabels failed'


  def openMergeLabelsModule(self):
    cliNode = self.getCLINode(slicer.modules.changelabel)
    parameters = self.mergeLabelsParameters()
    slicer.cli.setNodeParameters(cliNode, parameters)

    self.openModule('ChangeLabel')

  # 2) Model Maker
  def openExtractPage(self):
    if self.get('BoneModelMakerOutputNodeComboBox').currentNode() == None:
      self.get('BoneModelMakerOutputNodeComboBox').addNode()
    if self.get('SkinModelMakerOutputNodeComboBox').currentNode() == None:
      self.get('SkinModelMakerOutputNodeComboBox').addNode()

  #     a) Bone Model Maker
  def setupBoneModelMakerLabels(self):
    """ Update the labels of the bone model maker
    """
    labels = []
    labels.append(self.get('BoneLabelComboBox').currentColor)
    self.get('BoneModelMakerLabelsLineEdit').setText(', '.join(str(val) for val in labels))

  def boneModelMakerParameters(self):
    parameters = {}
    parameters["InputVolume"] = self.get('BoneModelMakerInputNodeComboBox').currentNode()
    parameters["ModelSceneFile"] = self.get('BoneModelMakerOutputNodeComboBox').currentNode()
    parameters["Labels"] = self.get('BoneModelMakerLabelsLineEdit').text
    parameters["Name"] = 'Bones'
    parameters['GenerateAll'] = False
    parameters["JointSmoothing"] = False
    parameters["SplitNormals"] = True
    parameters["PointNormals"] = True
    parameters["SkipUnNamed"] = True
    parameters["Decimate"] = 0.25
    parameters["Smooth"] = 10
    return parameters

  def runBoneModelMaker(self):
    cliNode = self.getCLINode(slicer.modules.modelmaker)
    parameters = self.boneModelMakerParameters()
    self.get('BoneModelMakerApplyPushButton').setChecked(True)
    cliNode = slicer.cli.run(slicer.modules.modelmaker, cliNode, parameters, wait_for_completion = True)
    self.get('BoneModelMakerApplyPushButton').setChecked(False)
    if cliNode.GetStatusString() == 'Completed':
      print 'Bone ModelMaker completed'
      self.resetCamera()
    else:
      print 'ModelMaker failed'

  def openModelsModule(self):
    self.openModule('Models')

  def openBoneModelMakerModule(self):
    cliNode = self.getCLINode(slicer.modules.modelmaker)
    parameters = self.boneModelMakerParameters()
    slicer.cli.setNodeParameters(cliNode, parameters)
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

  def skinModelMakerParameters(self):
    parameters = {}
    parameters["InputVolume"] = self.get('SkinModelMakerInputNodeComboBox').currentNode()
    parameters["OutputGeometry"] = self.get('SkinModelMakerOutputNodeComboBox').currentNode()
    parameters["Threshold"] = self.get('SkinModelMakerThresholdSpinBox').value + 0.1
    #parameters["SplitNormals"] = True
    #parameters["PointNormals"] = True
    #parameters["Decimate"] = 0.25
    parameters["Smooth"] = 10
    return parameters

  def runSkinModelMaker(self):
    cliNode = self.getCLINode(slicer.modules.grayscalemodelmaker)
    parameters = self.skinModelMakerParameters()
    self.get('SkinModelMakerApplyPushButton').setChecked(True)
    cliNode = slicer.cli.run(slicer.modules.grayscalemodelmaker, cliNode, parameters, wait_for_completion = True)
    self.get('SkinModelMakerApplyPushButton').setChecked(False)
    if cliNode.GetStatusString() == 'Completed':
      print 'Skin ModelMaker completed'
      # Set opacity
      newNode = self.get('SkinModelMakerOutputNodeComboBox').currentNode()
      newNodeDisplayNode = newNode.GetModelDisplayNode()
      newNodeDisplayNode.SetOpacity(0.2)

      # Set color
      colorNode = self.get('SkinModelMakerInputNodeComboBox').currentNode().GetDisplayNode().GetColorNode()
      color = [0, 0, 0]
      lookupTable = colorNode.GetLookupTable().GetColor(self.get('SkinLabelComboBox').currentColor, color)
      newNodeDisplayNode.SetColor(color)

      # Set Clip intersection ON
      newNodeDisplayNode.SetSliceIntersectionVisibility(1)

      # Reset camera
      self.resetCamera()
    else:
      print 'Skin ModelMaker failed'

  def openSkinModelMakerModule(self):
    cliNode = self.getCLINode(slicer.modules.grayscalemodelmaker)
    parameters = self.skinModelMakerParameters()
    slicer.cli.setNodeParameters(cliNode, parameters)

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
    self.removeObservers(self.updateVolumeRender)
    if volumeNode == None:
      return
    displayNode = volumeNode.GetNthDisplayNodeByClass(0, 'vtkMRMLVolumeRenderingDisplayNode')
    visible = False
    if displayNode != None:
      visible = displayNode.GetVisibility()
    self.get('VolumeRenderCheckBox').setChecked(visible)
    self.setupVolumeRenderLabels()
    self.addObserver(volumeNode, 'ModifiedEvent', self.updateVolumeRender)

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

  # 3) Create rest armature
  def openCreateArmaturePage(self):
    self.resetCamera()
    # Switch to 3D View only
    manager = slicer.app.layoutManager()
    manager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)

  # 3.A) Armature
  def loadArmatureFile(self):
    manager = slicer.app.ioManager()
    manager.openAddSceneDialog()

  def openArmaturesModule(self):
    # Finaly open armature module
    self.openModule('Armatures')

  # 4) Armatures Weight and Bones
  def openSkinningPage(self):
    activeArmatureNode = slicer.modules.armatures.logic().GetActiveArmature()
    if activeArmatureNode != None:
      self.get('SegmentBonesAmartureNodeComboBox').setCurrentNode(activeArmatureNode.GetAssociatedNode())

  #  a) Armatures Bones
  def segmentBonesParameters(self):
    parameters = {}
    parameters["RestLabelmap"] = self.get('SegmentBonesInputVolumeNodeComboBox').currentNode()
    parameters["ArmaturePoly"] = self.get('SegmentBonesAmartureNodeComboBox').currentNode()
    parameters["BodyPartition"] = self.get('SegmentBonesOutputVolumeNodeComboBox').currentNode()
    #parameters["Padding"] = 1
    #parameters["Debug"] = False
    #parameters["ArmatureInRAS"] = False
    return parameters

  def runSegmentBones(self):
    cliNode = self.getCLINode(slicer.modules.armaturebones)
    parameters = self.segmentBonesParameters()
    self.get('SegmentBonesApplyPushButton').setChecked(True)
    cliNode = slicer.cli.run(slicer.modules.armaturebones, cliNode, parameters, wait_for_completion = True)
    self.get('SegmentBonesApplyPushButton').setChecked(False)

    if cliNode.GetStatusString() == 'Completed':
      print 'Armature Bones completed'
    else:
      print 'Armature Bones failed'

  def openSegmentBonesModule(self):
    cliNode = self.getCLINode(slicer.modules.armaturebones)
    parameters = self.segmentBonesParameters()
    slicer.cli.setNodeParameters(cliNode, parameters)

    self.openModule('ArmatureBones')

    #  b) Armature Weight
  def armatureWeightParameters(self):
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
    return parameters

  def runArmatureWeight(self):
    cliNode = self.getCLINode(slicer.modules.armatureweight)
    parameters = self.armatureWeightParameters()
    self.get('ArmatureWeightApplyPushButton').setChecked(True)
    cliNode = slicer.cli.run(slicer.modules.armatureweight, cliNode, parameters, wait_for_completion = True)
    self.get('ArmatureWeightApplyPushButton').setChecked(False)

    if cliNode.GetStatusString() == 'Completed':
      print 'Armature Weight completed'
    else:
      print 'Armature Weight failed'

  def openArmatureWeightModule(self):
    cliNode = self.getCLINode(slicer.modules.armatureweight)
    parameters = self.armatureWeightParameters()
    slicer.cli.setNodeParameters(cliNode, parameters)

    self.openModule('ArmatureWeight')

  # 5) (Pose) Armature And Pose Body
  def openPoseArmaturePage(self):
    armatureLogic = slicer.modules.armatures.logic()
    if armatureLogic != None:
      armatureLogic.SetActiveArmatureWidgetState(3) # 3 is Pose

  # a) Eval Weight
  def evalWeightParameters(self):
    parameters = {}
    parameters["InputSurface"] = self.get('EvalWeightInputNodeComboBox').currentNode()
    parameters["OutputSurface"] = self.get('EvalWeightOutputNodeComboBox').currentNode()
    parameters["WeightDirectory"] = str(self.get('EvalWeightWeightDirectoryButton').directory)
    #parameters["IsSurfaceInRAS"] = False
    #parameters["PrintDebug"] = False
    return parameters

  def runEvalWeight(self):
    cliNode = self.getCLINode(slicer.modules.evalweight)
    parameters = self.evalWeightParameters()
    self.get('EvalWeightApplyPushButton').setChecked(True)
    cliNode = slicer.cli.run(slicer.modules.evalweight, cliNode, parameters, wait_for_completion = True)
    self.get('EvalWeightApplyPushButton').setChecked(False)

    if cliNode.GetStatusString() == 'Completed':
      print 'Evaluate Weight completed'
    else:
      print 'Evaluate Weight failed'

  def openEvalWeight(self):
    cliNode = self.getCLINode(slicer.modules.evalweight)
    parameters = self.evalWeightParameters()
    slicer.cli.setNodeParameters(cliNode, parameters)

    self.openModule('EvalWeight')

  # b) (Pose) Armatures
  def openPosedArmatureModule(self):
    self.openModule('Armatures')

  # c) Pose Body
  def poseBodyParameterChanged(self):
    cliNode = self.getCLINode(slicer.modules.posebody)
    if cliNode.IsBusy() == True:
      cliNode.Cancel()

  def poseBodyParameters(self):
    # Setup CLI node on input changed or apply changed
    parameters = {}
    parameters["ArmaturePoly"] = self.get('PoseBodyArmatureInputNodeComboBox').currentNode()
    parameters["SurfaceInput"] = self.get('PoseBodySurfaceInputComboBox').currentNode()
    parameters["WeightDirectory"] = str(self.get('PoseBodyWeightInputDirectoryButton').directory)
    parameters["OutputSurface"] = self.get('PoseBodySurfaceOutputNodeComboBox').currentNode()
    parameters["IsSurfaceInRAS"] = False
    parameters["IsArmatureInRAS"] = False
    parameters["LinearBlend"] = True
    return parameters

  def runPoseBody(self):
    cliNode = self.getCLINode(slicer.modules.posebody)
    parameters = self.poseBodyParameters()
    slicer.cli.setNodeParameters(cliNode, parameters)
    cliNode.SetAutoRunMode(cliNode.AutoRunOnAnyInputEvent)

    if self.get('PoseBodyApplyPushButton').checkState == False:
      if cliNode.IsBusy() == False:
        self.get('PoseBodyApplyPushButton').setChecked(True)
        slicer.modules.posebody.logic().ApplyAndWait(cliNode)
        self.get('PoseBodyApplyPushButton').setChecked(False)
      else:
        cliNode.Cancel()
    else:
      cliNode.SetAutoRun(self.get('PoseBodyApplyPushButton').isChecked())

  def openPoseBodyModule(self):
    cliNode = self.getCLINode(slicer.modules.posebody)
    parameters = self.poseBodyParameterss()
    slicer.cli.setNodeParameters(cliNode, parameters)

    self.openModule('PoseBody')

  def setWeightDirectory(self, dir):
    if self.get('EvalWeightWeightDirectoryButton').directory != dir:
      self.get('EvalWeightWeightDirectoryButton').directory = dir

    if self.get('PoseBodyWeightInputDirectoryButton').directory != dir:
      self.get('PoseBodyWeightInputDirectoryButton').directory = dir

    if self.get('PoseLabelmapWeightDirectoryButton').directory != dir:
      self.get('PoseLabelmapWeightDirectoryButton').directory = dir

  def createOutputSurface(self, node):
    if node == None:
      return

    if self.get('PoseBodySurfaceOutputNodeComboBox').currentNode() != None:
      return

    newNode = self.get('PoseBodySurfaceOutputNodeComboBox').addNode()
    newNode.SetName('%s-posed' % node.GetName())

  # 6) Resample NOTE: SHOULD BE LAST STEP
  def openPoseLabelmapPage(self):
    pass

  def poseLabelmapParameters(self):
    parameters = {}
    parameters["RestLabelmap"] = self.get('PoseLabelmapInputNodeComboBox').currentNode()
    parameters["ArmaturePoly"] = self.get('PoseLabelmapArmatureNodeComboBox').currentNode()
    parameters["WeightDirectory"] = str(self.get('PoseLabelmapWeightDirectoryButton').directory)
    parameters["PosedLabelmap"] = self.get('PoseLabelmapOutputNodeComboBox').currentNode()
    parameters["LinearBlend"] = True
    #parameters["MaximumPass"] = 4
    #parameters["Debug"] = False
    #parameters["IsArmatureInRAS"] = False
    return parameters

  def runPoseLabelmap(self):
    cliNode = self.getCLINode(slicer.modules.poselabelmap)
    parameters = self.poseLabelmapParameters()
    self.get('PoseLabelmapApplyPushButton').setChecked(True)
    cliNode = slicer.cli.run(slicer.modules.poselabelmap, cliNode, parameters, wait_for_completion = True)
    self.get('PoseLabelmapApplyPushButton').setChecked(False)
    if cliNode.GetStatusString() == 'Completed':
      print 'Pose Labelmap completed'
    else:
      print 'Pose Labelmap failed'

  def openPoseLabelmap(self):
    cliNode = self.getCLINode(slicer.modules.poselabelmap)
    parameters = self.poseLabelmapParameters()
    slicer.cli.setNodeParameters(cliNode, parameters)

    self.openModule('PoseLabelmap')

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

  def removeObservers(self, method):
    for object, event, method, group, tag in self.Observations:
      if method == method:
        object.RemoveObserver(tag)

  def addObserver(self, object, event, method, group = 'none'):
    if self.hasObserver(object, event, method):
      return
    tag = object.AddObserver(event, method)
    self.Observations.append([object, event, method, group, tag])

  def hasObserver(self, object, event, method):
    for o, e, m, g, t in self.Observations:
      if o == object and e == event and m == method:
        return True
    return False

  def getCLINode(self, cliModule):
    """ Return the cli node to use for a given CLI module. Create the node in
    scene if needed.
    """
    cliNode = slicer.mrmlScene.GetFirstNodeByName(cliModule.title)
    if cliNode == None:
      cliNode = slicer.cli.createNode(cliModule)
      cliNode.SetName(cliModule.title)
    return cliNode

  def resetCamera(self):
    # Reset focal view around volumes
    manager = slicer.app.layoutManager()
    for i in range(0, manager.threeDViewCount):
      manager.threeDWidget(i).threeDView().resetFocalPoint()

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
