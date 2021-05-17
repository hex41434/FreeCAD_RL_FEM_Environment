import os
import sys
import FreeCAD
import Part
import ObjectsFem
import Mesh
import MeshPart
import Fem
import femmesh.femmesh2mesh
from femtools import ccxtools
import trimesh
import numpy as np
import openpyxl
from openpyxl import Workbook

def init_shape(doc, load_3d=True):

    if load_3d:
        # pth = "./savedmesh/"
        pth = "/scratch/aifa/DATA/FCAD_3D/"
        # filename = "_20210517153229478077.obj"#init
        filename = "_20210517163105503117.obj" 
        
        Mesh_obj_Name = filename.replace(".obj","")

        Mesh.insert(os.path.join(pth,filename),doc.Name)
        print("\n ******* Mesh.obj is loaded ******* \n")
        doc.addObject("Part::Feature","initShape")
        
        msh = doc.getObject(Mesh_obj_Name)
        mesh_OK = checkMesh(msh.Mesh)

        err = False
        if not mesh_OK:
            err = True
            print("ERRRR")

        s=Part.Shape()
        s.makeShapeFromMesh(doc.getObject(Mesh_obj_Name).Mesh.Topology,0.100000)
        doc.getObject("initShape").Shape=s
        doc.getObject("initShape").purgeTouched()
        del s

        sh=doc.initShape.Shape
        sh=Part.Solid(sh)
        obj=doc.addObject("Part::Feature","Solid")
        obj.Shape=sh
        del sh, obj
        
        state0_trimesh = Meshobj_to_trimesh(doc.getObject(Mesh_obj_Name).Mesh)
        doc = remove_old_mesh_result_mesh(doc,name=Mesh_obj_Name)
        doc = remove_old_shape_result_mesh(doc,name="initShape")
        
    else:        
        Solid_obj = doc.addObject("Part::Box", "Solid")
        Solid_obj.Height = 2
        Solid_obj.Width = 5
        Solid_obj.Length = 100
        
        Mesh_obj_Name = "initBox"

        msh = doc.addObject("Mesh::Feature",Mesh_obj_Name)
        prt = doc.getObject("Solid")
        shp = prt.Shape.copy(False)
        shp.Placement = prt.getGlobalPlacement()
        msh.Mesh=MeshPart.meshFromShape(Shape=shp,Fineness=2,SecondOrder=0,Optimize=1,AllowQuad=0)
        del prt, shp

        state0_trimesh = Meshobj_to_trimesh(doc.getObject(Mesh_obj_Name).Mesh)
        doc = remove_old_shape_result_mesh(doc,name=Mesh_obj_Name)

    return doc,state0_trimesh,Mesh_obj_Name

def checkMesh(msh):
    mesh_OK = True

    invalid_points = Mesh.Mesh.hasInvalidPoints(msh)
    has_manifolds = Mesh.Mesh.hasNonManifolds(msh)
    orient_faces = Mesh.Mesh.hasNonUniformOrientedFacets(msh)
    self_intersect = Mesh.Mesh.hasSelfIntersections(msh)

    print("--------")
    print(f"invalid_points: {invalid_points}, has_manifolds: {has_manifolds},orient_faces: {orient_faces}, self_intersect: {self_intersect}")
    print("--------")
    
    if (invalid_points) or (has_manifolds) or (orient_faces) or (self_intersect):
        mesh_OK = False
        print("     XXX Mesh is not healthy!  XXX ")

    return mesh_OK

# def add_to_excel_file(initMesh,force_position,force_val,sav):
def add_to_excel_file(rows_list):

    # pth = "/scratch/aifa/DATA/FCAD_3D"
    pth = "/scratch/aifa/MyRepo/RL_FEM/FreeCAD_RL_FEM_Environment/scripts/savedmesh"
    xls_obj = os.path.join(pth,'demo.xlsx')
    
    # row_data = [f"{initMesh}.obj",force_position,force_val,f"{sav}.obj"]
    
    if not os.path.exists(xls_obj):
        wbook = Workbook()
    else:
        wbook = openpyxl.load_workbook(xls_obj)
        
    wsheet = wbook.worksheets[0]
    for row_data in rows_list:
        wsheet.append(row_data)    
    
    wbook.save(xls_obj)

def Add_Ref_Object(doc):
    box_obj = doc.addObject('Part::Box', 'RefBox')
    box_obj.Height = 2
    box_obj.Width = 5
    box_obj.Length = 100
    return doc

def document_setup(doc,trimesh_scene_meshes):
    doc = Add_Ref_Object(doc)

    #state 0
    doc, state0, initMesh = init_shape(doc)
    trimesh_scene_meshes.append(state0)

    doc = create_analysis(doc)
    doc = add_solver(doc)
    doc = add_material(doc)

    doc = create_constraint_fixed(doc)
    doc = create_constraint_force(doc)
    return doc,trimesh_scene_meshes,initMesh

def create_analysis(doc):
    analysis_object = ObjectsFem.makeAnalysis(doc, 'Analysis')
    return doc

def add_solver(doc):
    doc.Analysis.addObject(ObjectsFem.makeSolverCalculixCcxTools(doc))
    return doc

def create_shape_femmesh(doc):
    fem_ok = True
    # femmesh_obj = ObjectsFem.makeMeshNetgen(doc, 'FEMMeshNetgen')
    femmesh_obj = doc.addObject('Fem::FemMeshShapeNetgenObject', 'FEMMeshNetgen')
    femmesh_obj.Fineness = "VeryCoarse"
    #femmesh_obj.SecondOrder = False
    femmesh_obj.Shape = doc.Solid
    try:
        doc.recompute()
    except:
        print("GZ")
    if len(femmesh_obj.FemMesh.Faces)>0:
        doc.Analysis.addObject(femmesh_obj)
    else:
        fem_ok = False
        print("ERRR - no femmesh is created!")    
    return doc, fem_ok

def create_constraint_fixed(doc):
    doc.addObject("Fem::ConstraintFixed","FemConstraintFixed")
    doc.FemConstraintFixed.Scale = 1
    doc.FemConstraintFixed.References = [(doc.RefBox,"Face2"), (doc.RefBox,"Face1")]
    doc.Analysis.addObject(doc.FemConstraintFixed)
    doc.recompute()
    return doc

def create_constraint_force(doc):
    doc.addObject("Fem::ConstraintForce","FemConstraintForce")
    doc.FemConstraintForce.Force = 8000
    doc.FemConstraintForce.Direction = (doc.RefBox,["Face6"])# **** direction Z axis
    doc.FemConstraintForce.Reversed = False
    doc.FemConstraintForce.Scale = 1
    doc.FemConstraintForce.References = [(doc.RefBox,"Face6")]
    doc.Analysis.addObject(doc.FemConstraintForce)
    doc.recompute()
    return doc

def add_material(doc):
    material_object = ObjectsFem.makeMaterialSolid(doc, "SolidMaterial")
    mat = material_object.Material
    mat['Name'] = "Steel-Generic"
    mat['YoungsModulus'] = "210000 MPa"
    mat['PoissonRatio'] = "0.30"
    mat['Density'] = "7900 kg/m^3"
    material_object.Material = mat
    doc.Analysis.addObject(material_object)
    return doc

def run_analysis(doc):

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
        fem_volume=True
        print("Analysis done successfully...")
    else:
        fem_volume=False
        print("problem occurred! {}\n".format(message))  # in python console
    return doc, fem_volume

def create_mesh_from_result(doc):
    
    out_mesh = []
    femmesh_result = doc.getObject("ResultMesh").FemMesh
    ccx_result = doc.getObject("CCX_Results")
    try:
        out = femmesh.femmesh2mesh.femmesh_2_mesh(femmesh_result, ccx_result)
        out_mesh = Mesh.Mesh(out)
        Mesh.show(out_mesh)
        print("mesh_out is converted to mesh...")
        doc.recompute()
    except:
        print("result can not be converted to mesh!")
        
    return doc, out_mesh
    
def export_result_mesh(doc,indx):
    meshobj = [] 
    meshobj.append(doc.getObject("Mesh"))
 
    savepath = "/scratch/aifa/DATA/FCAD_3D"
    savepath = os.path.join(os.getcwd(),savepath)
    filename = f"resultmesh_fcad{indx}.obj"

    Mesh.export(meshobj, os.path.join(savepath, filename))
    print("result mesh is exported as .obj file ... ")
    
def set_trimesh_color(mytrimesh):
    rand_color = trimesh.visual.color.random_color()    
    
    cv = trimesh.visual.color.ColorVisuals(mesh=mytrimesh, face_colors=rand_color, vertex_colors=None)
    mytrimesh.visual = cv
    return mytrimesh

def Meshobj_to_trimesh(mesh_obj):
    v,f = mesh_obj.Topology
    mytrimesh = trimesh.Trimesh(vertices=v,faces=f)

    return mytrimesh
 
def remove_old_femmesh(doc):
    femmesh_list = doc.findObjects("Fem::FemMeshShapeNetgenObject")
    if femmesh_list:
        doc.removeObject(femmesh_list[0].Name)
        doc.recompute()
        print("old femmesh is removed...")
    return doc

def create_shape_from_mesh(doc,mesh_topology):    
    
    doc.addObject("Part::Feature","Mesh001")
    shape = Part.Shape()
    shape.makeShapeFromMesh(mesh_topology,0.100000)
    doc.getObject("Mesh001").Shape = shape
    doc.getObject("Mesh001").purgeTouched()
    
    print("shape from mesh done...")
    return doc

def convert_to_solid(doc):
        
    shp = doc.Mesh001.Shape
    shp = Part.Solid(shp)
    solid = doc.addObject("Part::Feature","Solid")
    solid.Shape = shp

    doc.recompute()
    return doc, solid

def set_constraint_force_placement(doc,force_position,region=1,force_normal=None):

    solid = doc.Solid
    ref_list = []
    ref_indx = []
    
    if len(solid.Shape.Faces)==6:
        ref_list.append((solid, f"Face6"))
        ref_indx.append(5) 
    else:
        length_min = solid.Shape.BoundBox.XMin
        length_max = solid.Shape.BoundBox.XMax
        safety = 2

        start = length_min+safety if force_position<length_min+safety else (force_position - region)
        end = length_max-safety if force_position>length_max-safety else (force_position + region)

        vec = FreeCAD.Base.Vector((0,0,1))
        for i in range(len(solid.Shape.Faces)):
            find_x = solid.Shape.Faces[i].CenterOfMass.x
            find_z = solid.Shape.Faces[i].CenterOfMass.z

            if find_x <=end and find_x>=start : 
                find_face = solid.Shape.Faces[i]
                u,v = find_face.Surface.parameter(find_face.CenterOfMass)
                nrml = find_face.normalAt(u,v)
                if nrml.dot(vec) >= 0.75:
                    ref_list.append((solid, f"Face{i+1}"))
                    ref_indx.append(i) 

    print(f'num of faces as force constraints: {len(ref_list)}')
    doc.FemConstraintForce.References  = ref_list
    #set
    doc.FemConstraintForce.Direction = (doc.RefBox,["Face6"])
    doc.FemConstraintForce.Reversed = False
    doc.FemConstraintForce.Scale = 1
    doc.recompute()
    
    return doc, ref_indx

def set_constraint_force_value(doc, force):    
    doc.FemConstraintForce.Force = force
    return doc

def set_constraint_fixed(doc):
    
    solid = doc.Solid
    vec = FreeCAD.Base.Vector((1,0,0))
    ref_list = []
    ref_indx = []
    
    for i in range(len(solid.Shape.Faces)):
        find_x = solid.Shape.Faces[i].CenterOfMass.x
        if find_x <=1.2: # condition for region selection

            find_face = solid.Shape.Faces[i]
            u,v = find_face.Surface.parameter(find_face.CenterOfMass)
            nrml = find_face.normalAt(u,v)

            if np.abs(nrml.dot(vec)) >=0.99:
                ref_list.append((solid, f"Face{i+1}"))
                ref_indx.append(i)

    vec = FreeCAD.Base.Vector((-1,0,0))
    for i in range(len(solid.Shape.Faces)):
        find_x = solid.Shape.Faces[i].CenterOfMass.x
        if find_x >=48.5: # condition for region selection

            find_face = solid.Shape.Faces[i]
            u,v = find_face.Surface.parameter(find_face.CenterOfMass)
            nrml = find_face.normalAt(u,v)

            if np.abs(nrml.dot(vec)) >=0.99:
                ref_list.append((solid, f"Face{i+1}"))
                ref_indx.append(i)

    doc.FemConstraintFixed.References  = ref_list
    print(f'num of faces as fixed constraints: {len(ref_list)}')

    return doc, ref_indx

def generate_complete_meshes_scene(doc,trimesh_scene_meshes):
    
    x,y,z = trimesh_scene_meshes[0].center_mass
 #     x,y,z = np.array(solid.Shape.BoundBox.Center)
        
    camera_new_pose = np.array([
            [ 0.70710678,  0.0        ,  0.70710678,     x],
            [ 0.70710678,  0.0        ,  0.70710678,     y+60],
            [ 0.0       , -1.0        ,  0.0       ,     z],
            [ 0.0       ,  0.0        ,  0.0       ,     1.0]
            ])

    mesh_scene = trimesh.Scene(trimesh_scene_meshes)
    mesh_scene.camera_transform = camera_new_pose
    return mesh_scene

def clear_constraints(doc):
    doc.FemConstraintFixed.References = [(doc.RefBox,["Face6"])]
    doc.FemConstraintForce.References = [(doc.RefBox,["Face6"])]
    doc.FemConstraintForce.Direction = None
    
    doc.recompute()
    print("FemConstraints cleared...")
    
    return doc

def remove_old_mesh_result_mesh(doc,name="Mesh"):
    mesh_result_mesh = doc.getObject(name)
    if mesh_result_mesh:
        doc.removeObject(mesh_result_mesh.Name)
        print("mesh result mesh is removed...")
    return doc 
    

def remove_old_shape_result_mesh(doc,name="Mesh001"):
    shape_result_mesh = doc.getObject(name)
    if shape_result_mesh:
        doc.removeObject(shape_result_mesh.Name)
        print("shape result mesh is removed...")
    return doc

def remove_old_solid(doc):
    solid_result_mesh = doc.getObject("Solid")
    if solid_result_mesh: 
        doc.removeObject(solid_result_mesh.Name)
        print("old solid result mesh is removed...")
    else:
        print("there is no solid object to be removed... ")
    
    doc.recompute()    
    return doc

def add_constraints_to_scene_meshes(doc,constraint_indx,color,view_markers=True):
    
    _color = [255,0,0,0] if color=='red' else [0,0,255,0]
    
    N = len(constraint_indx)
    if view_markers:# view the selected faces by marker
        meshes=[]
        for k in range(N):
            center = doc.Solid.Shape.Faces[constraint_indx[k]].CenterOfMass        

            marker = trimesh.creation.box(extents=(0.5,0.5,0.5))
            marker.apply_translation(np.array(center))
            marker.visual = trimesh.visual.color.ColorVisuals(mesh=marker, face_colors=_color)            
            meshes.append(marker)
            
        constraint_scene = trimesh.util.concatenate(meshes)

    else:#color the selected faces(not all the faces are shown)
        VEC = np.zeros((N*3,3))
        FAC = np.arange(N*3).reshape(-1,3)
        
        for k in range(N):    
            ind = k*3
            _ver0 = np.array(doc.Solid.Shape.Faces[constraint_indx[k]].Vertexes[0].Point)
            _ver1 = np.array(doc.Solid.Shape.Faces[constraint_indx[k]].Vertexes[1].Point)
            _ver2 = np.array(doc.Solid.Shape.Faces[constraint_indx[k]].Vertexes[2].Point)

            VEC[ind]   = _ver0
            VEC[ind+1] = _ver1
            VEC[ind+2] = _ver2
        
        constraint_scene = trimesh.Trimesh(vertices=VEC, faces=FAC)
        
    return constraint_scene

def trimesh_to_mesh_topology(result_trimesh):
    vecs = [] 
    fcs = [] 

    vs = result_trimesh.vertices
    fs = result_trimesh.faces

    for i in range(len(vs)):
        v = FreeCAD.Base.Vector(vs[i])
        vecs.append(v)

    for i in range(len(fs)):
        f = (fs[i][0],fs[i][1],fs[i][2])
        fcs.append(f)

    result_trimesh_topology = (vecs , fcs)
    return result_trimesh_topology

def save_trimesh(mytrimesh, saved_filename):
    pth = "/scratch/aifa/DATA/FCAD_3D"
    file_obj = os.path.join(pth,saved_filename)
    mytrimesh.export(file_obj)
    print(f'file is saved: {file_obj}')

def scene_view_normals(mytrimesh):
    vec = np.column_stack((mytrimesh.vertices, mytrimesh.vertices + (mytrimesh.vertex_normals * mytrimesh.scale * .05)))
    vec = vec.reshape((-1, 2, 3))

    path = trimesh.load_path(vec)
    path.colors = np.row_stack((len(vec)*[0,0,0,255])).reshape((-1,4))
    scene = trimesh.Scene([mytrimesh, path])
    return scene

def set_camera_pose(doc, scene):
    
    x,y,z = doc.RefBox.Shape.CenterOfMass
    camera_new_pose = np.array([
            [ 0.70710678,  0.0        ,  0.70710678,     x],
            [ 0.70710678,  0.0        ,  0.70710678,     y+80],
            [ 0.0       , -1.0        ,  0.0       ,     z],
            [ 0.0       ,  0.0        ,  0.0       ,     1.0]
            ])
    scene.camera_transform = camera_new_pose
    return scene
    

def concat_meshes(mesh1,mesh2):
    concatenated_mesh = trimesh.util.concatenate(mesh1,mesh2)
    return concatenated_mesh

