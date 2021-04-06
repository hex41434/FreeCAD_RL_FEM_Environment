# freecad Macro - create shape and calculate deformations and save ply
import FreeCAD
import FreeCADGui
import Part
import ObjectsFem
import Fem
import Mesh
import femmesh.femmesh2mesh
from femtools import ccxtools

doc = App.newDocument("doc")

# Gui.activeDocument().activeView().viewIsometric()
# Gui.SendMsgToActiveView("ViewFit")

box_obj = doc.addObject('Part::Box', 'Box')
box_obj.Height = 2
box_obj.Width = 5
box_obj.Length = 50

# Gui.activeDocument().activeView().viewIsometric()
# Gui.SendMsgToActiveView("ViewFit")

analysis_object = ObjectsFem.makeAnalysis(FreeCAD.ActiveDocument, 'Analysis')

femmesh_obj = doc.addObject('Fem::FemMeshShapeNetgenObject', 'CubeFemMesh')
femmesh_obj.Shape = box_obj 
femmesh_obj.MaxSize = 1

doc.Analysis.addObject(femmesh_obj)
# FemGui.setActiveAnalysis(App.activeDocument().Analysis)
doc.recompute()

doc.addObject("Fem::ConstraintFixed","FemConstraintFixed")
doc.FemConstraintFixed.Scale = 1
doc.Analysis.addObject(doc.FemConstraintFixed)
for amesh in doc.Objects:
    if "FemConstraintFixed" == amesh.Name:
        print('amesh.ViewObject.Visibility = True')
    elif "Mesh" in amesh.TypeId:
        aparttoshow = amesh.Name.replace("_Mesh","")
        for apart in doc.Objects:
            if aparttoshow == apart.Name:
                print('apart.ViewObject.Visibility = True')
        print('amesh.ViewObject.Visibility = False')

doc.recompute()

doc.FemConstraintFixed.Scale = 1
doc.FemConstraintFixed.References = [(doc.Box,"Face2")]
doc.recompute()

doc.addObject("Fem::ConstraintForce","FemConstraintForce")
doc.FemConstraintForce.Force = 1.0
doc.FemConstraintForce.Reversed = False
doc.FemConstraintForce.Scale = 1
doc.Analysis.addObject(doc.FemConstraintForce)

for amesh in doc.Objects:
    if "FemConstraintForce" == amesh.Name:
        print("amesh.ViewObject.Visibility = True")
    elif "Mesh" in amesh.TypeId:
        aparttoshow = amesh.Name.replace("_Mesh","")
        for apart in doc.Objects:
            if aparttoshow == apart.Name:
                print("apart.ViewObject.Visibility = True")
        print("amesh.ViewObject.Visibility = False")

doc.recompute()
doc.FemConstraintForce.Force = 1000
doc.FemConstraintForce.Direction = None
doc.FemConstraintForce.Reversed = True
doc.FemConstraintForce.Scale = 1
doc.FemConstraintForce.References = [(doc.Box,"Face6")]
doc.recompute()

# FemGui.getActiveAnalysis().addObject(ObjectsFem.makeSolverCalculixCcxTools(doc))
analysis_object.addObject(ObjectsFem.makeSolverCalculixCcxTools(doc))

## material
material_object = ObjectsFem.makeMaterialSolid(doc, "SolidMaterial")
mat = material_object.Material
mat['Name'] = "Steel-Generic"
mat['YoungsModulus'] = "210000 MPa"
mat['PoissonRatio'] = "0.30"
mat['Density'] = "7900 kg/m^3"
material_object.Material = mat
# FemGui.getActiveAnalysis().addObject(material_object)
analysis_object.addObject(material_object)

# run the analysis step by step
fea = ccxtools.FemToolsCcx()
fea.update_objects()
fea.setup_working_dir()
fea.setup_ccx()
message = fea.check_prerequisites()
if not message:
    fea.purge_results()
    fea.write_inp_file()
    fea.ccx_run()
    fea.load_results()
else:
    FreeCAD.Console.PrintError("Oh, we have a problem! {}\n".format(message))  # in report view
    print("Oh, we have a problem! {}\n".format(message))  # in python console

# # show some results
# for m in analysis_object.Group:
#     if m.isDerivedFrom('Fem::FemResultObject'):
#         result_object = m
#         break
# femmesh_obj = doc.getObject("FEMMeshGmsh")

femmesh_obj = doc.getObject("ResultMesh").FemMesh
result = doc.getObject("CCX_Results")

out_mesh = femmesh.femmesh2mesh.femmesh_2_mesh(femmesh_obj, result)
Mesh.show(Mesh.Mesh(out_mesh))

#export one mesh as .ply file
__objs__=[]
__objs__.append(doc.getObject("Mesh"))

path ="C:/Users/Aida/Documents/FreeCAD_Macros/" 

filename = path + "custom" + ".ply"

Mesh.export(__objs__,filename)
del __objs__




