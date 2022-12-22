import os
import sys

import pymeshlab
conda_active_path = sys.exec_prefix
# conda_active_path = os.environ["CONDA_PREFIX"]
sys.path.append(os.path.join(conda_active_path, "lib"))

import FreeCAD
import Part
import ObjectsFem
import Mesh
import MeshPart
import Fem
import femmesh.femmesh2mesh
try:
    from femtools import ccxtools
except ImportError:
    print('ccxtools_import_passed...')
from femtools import ccxtools
from femmesh.gmshtools import GmshTools as gt
import trimesh
import numpy as np
import datetime
import random
import openpyxl
from openpyxl import Workbook
import yaml


class FreeCADWrapper(object):
    """Class allowing to interact with the FreeCad simulation.

    TODO: more description
    """

    def __init__(self, **kwargs):
        """
        Args:
            path: path to the .ply file.
        """
        env_config_path = './env_config.yaml'
        with open(env_config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        allowed_keys = {
            'save_path', 
            'loaded_mesh_path',
            'loaded_mesh_filename',
            'gt_mesh_path',
            'view_meshes',
            'xls_pth',
            'xls_filename'}

        self.save_path = cfg['save_path']
        self.loaded_mesh_path = cfg['loaded_mesh_path']
        self.loaded_mesh_filename = cfg['loaded_mesh_filename']
        self.view_meshes = cfg['view_meshes']
        self.xls_pth = cfg['xls_pth']
        self.xls_filename = cfg['xls_filename']
        self.gt_mesh_path = cfg['gt_mesh_path']

        self.__dict__.update((k, v) for k, v in kwargs.items() if (k in allowed_keys) and not (v is None))


        self.doc = FreeCAD.newDocument('doc')
        self.trimesh_scene_meshes = []
        self.constraint_scene_meshes = []
        self.view_meshes = view_meshes #True # for jupyter nb 
        self.step_no = 0
        self.fail = False
        self.doc,self.state0_trimesh, self.initMesh_Name, self.mesh_OK = self.init_shape(self.load_3d)
        if not self.mesh_OK: print('*** init mesh is not acceptable!')
        self.trimesh_scene_meshes.append(self.state0_trimesh)

    def create_fem_analysis(self,action):

#         (self.force_position, self.force_val) = self.generate_action()
        (self.force_position, self.force_val) = action
        print(f"force position : {self.force_position}\n")

        self.doc , self.fem_ok = self.create_shape_femmesh()    
        if not self.fem_ok: print('*** self.fem_ok is False!')

        self.doc , self.fixed_indx = self.set_constraint_fixed()
        self.doc , self.force_indx = self.set_constraint_force_placement(
            force_position=self.force_position)    
                                                
        if len(self.fixed_indx)==0 or len(self.force_indx)==0:
            print("force or fixed constraints are empty... no more analysis will be executed... ")
        
        self.fixed_scene = self.add_constraints_to_scene_meshes(self.fixed_indx,color='blue')
        self.force_scene = self.add_constraints_to_scene_meshes(self.force_indx,color='red')
        self.constraint_scene_meshes.append(self.concat_meshes(self.fixed_scene,self.force_scene))
        self.doc = self.remove_old_shape_result_mesh()
        self.doc = self.remove_old_mesh_result_mesh()
        
        self.doc = self.set_constraint_force_value(self.force_val)

    def generate_action(self):
            
            # self.force_position = np.random.randint(20,80)
            self.force_position = np.random.randint(2,8)
            # self.force_val = 180000
            # self.force_val = random.randrange(2500, 10000, 500)
            self.force_val = random.randrange(250000, 300000, 500)#STEEL
            self.action = (self.force_position, self.force_val)
            return self.force_position, self.force_val
    
    def fem_step(self,mesh_decimation=True):

        self.doc, self.fem_volume = self.run_analysis()
        print(f".....................................fem_volume:{self.fem_volume}")
        if not self.fem_volume: print("fem_volume is not OK...")

        #create and export result mesh
        self.doc, self.out_mesh = self.create_mesh_from_result()
        if (not self.out_mesh) or (not self.checkMesh(self.out_mesh)): self.fail = True   

        self.result_trimesh = self.Meshobj_to_trimesh(self.out_mesh)
        if(mesh_decimation):
        # the mesh is needed to be simplified? 
            self.result_trimesh, self.trimesh_topology = self.simplify_trimesh_result(
                self.result_trimesh)
        else:
            self.trimesh_topology = self.trimesh_to_mesh_topology(
                self.result_trimesh)
            
        self.trimesh_scene_meshes.append(self.result_trimesh)
        
        self.save_state()
        self.prepare_for_next_fem_step()
        # self.list_doc_objects()

    def prepare_for_next_fem_step(self):
        self.doc = self.remove_old_femmesh()
        self.doc = self.clear_constraints()
        self.doc = self.remove_old_solid()

        self.doc = self.create_shape_from_mesh(self.trimesh_topology) 
        self.convert_to_solid()
        
    def convert_to_solid(self):        
        shp = self.doc.Mesh001.Shape
        shp = Part.Solid(shp)
        solid = self.doc.addObject("Part::Feature","Solid")
        solid.Shape = shp

        self.doc.recompute()

    def clear_constraints(self):
        self.doc.FemConstraintFixed.References = [(self.doc.RefBox,["Face6"])]
        self.doc.FemConstraintForce.References = [(self.doc.RefBox,["Face6"])]
        self.doc.FemConstraintForce.Direction = None
        
        self.doc.recompute()
        print("FemConstraints cleared...")
        return self.doc
    
    def remove_old_solid(self):
        solid_result_mesh = self.doc.getObject("Solid")
        if solid_result_mesh: 
            self.doc.removeObject(solid_result_mesh.Name)
            print("old solid result mesh is removed...")
        else:
            print("there is no solid object to be removed... ")
        
        self.doc.recompute()    
        return self.doc
        
    
    def create_shape_from_mesh(self,mesh_topology):    
        
        self.doc.addObject("Part::Feature","Mesh001")
        shape = Part.Shape()
        shape.makeShapeFromMesh(mesh_topology,0.100000)
        self.doc.getObject("Mesh001").Shape = shape
        self.doc.getObject("Mesh001").purgeTouched()
        
        print("shape from mesh done...")
        return self.doc

    def trimesh_to_mesh_topology(self,result_trimesh):
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
        
    def simplify_trimesh_result(self,trmesh):
        m = pymeshlab.Mesh(trmesh.vertices, trmesh.faces)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(m)
        ms.simplification_quadric_edge_collapse_decimation(
            targetperc=.6,
            preservenormal=True,
            preservetopology=True)
        
        m_ = ms.current_mesh()
        
        result_trimesh_decimated = trimesh.Trimesh(
            vertices=m_.vertex_matrix(),
            faces=m_.face_matrix())
        
        ms.delete_current_mesh()
        trimesh_decimated_topology = self.trimesh_to_mesh_topology(result_trimesh_decimated)
        result_trimesh_decimated = self.set_trimesh_color(result_trimesh_decimated)

        return result_trimesh_decimated,trimesh_decimated_topology

    def set_trimesh_color(self,mytrimesh):
        rand_color = trimesh.visual.color.random_color()    
        
        cv = trimesh.visual.color.ColorVisuals(
            mesh=mytrimesh, 
            face_colors=rand_color, 
            vertex_colors=None)
        
        mytrimesh.visual = cv
        return mytrimesh

    def run_analysis(self):

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
            self.fem_volume=True
            print("Analysis done successfully...")
        else:
            self.fem_volume=False
            print("problem occurred! {}\n".format(message))  # in python console
        return self.doc, self.fem_volume

    def create_mesh_from_result(self):
        out_mesh = []
        femmesh_result = self.doc.getObject("ResultMesh").FemMesh
        ccx_result = self.doc.getObject("CCX_Results")
        try:
            out = femmesh.femmesh2mesh.femmesh_2_mesh(femmesh_result, ccx_result)
            out_mesh = Mesh.Mesh(out)
            Mesh.show(out_mesh)
            print("mesh_out is converted to mesh...")
            self.doc.recompute()
        except:
            print("result can not be converted to mesh!")
            
        return self.doc, out_mesh
        
    def set_constraint_force_value(self, force):    
        self.doc.FemConstraintForce.Force = force
        return self.doc
        
    def set_constraint_fixed(self):
        solid = self.doc.Solid
        vec = FreeCAD.Base.Vector((1,0,0))
        ref_list = []
        ref_indx = []
        
        fix_heads = False
        if fix_heads:
            #fix the right side
            for i in range(len(solid.Shape.Faces)):
                find_x = solid.Shape.Faces[i].CenterOfMass.x
                if find_x <=1.2: # condition for region selection

                    find_face = solid.Shape.Faces[i]
                    u,v = find_face.Surface.parameter(find_face.CenterOfMass)
                    nrml = find_face.normalAt(u,v)
                    if len(ref_indx)<=15:
                        if np.abs(nrml.dot(vec)) >=0.99:
                            ref_list.append((solid, f"Face{i+1}"))
                            ref_indx.append(i)    
                    else:
                        break
            #fix the left side
            vec = FreeCAD.Base.Vector((-1,0,0))
            for i in range(len(solid.Shape.Faces)):
                find_x = solid.Shape.Faces[i].CenterOfMass.x
                if find_x >=48.5: # condition for region selection

                    find_face = solid.Shape.Faces[i]
                    u,v = find_face.Surface.parameter(find_face.CenterOfMass)
                    nrml = find_face.normalAt(u,v)
                    if len(ref_indx)<=30:
                        if np.abs(nrml.dot(vec)) >=0.99:
                            ref_list.append((solid, f"Face{i+1}"))
                            ref_indx.append(i)
                    else:
                        break
                    
        else: #fix from the bottom
            for i in range(len(solid.Shape.Faces)):
                find_z = solid.Shape.Faces[i].CenterOfMass.z
                if find_z <=0.1: # condition for region selection
                    find_face = solid.Shape.Faces[i]
                    ref_list.append((solid, f"Face{i+1}"))
                    ref_indx.append(i)
                                    
        self.doc.FemConstraintFixed.References  = ref_list
        print(f'num of faces as fixed constraints: {len(ref_list)}')

        return self.doc, ref_indx
        
    def set_constraint_force_placement(self,force_position,region=1,force_normal=None):

        solid = self.doc.Solid
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
                # find_z = solid.Shape.Faces[i].CenterOfMass.z

                if find_x <=end and find_x>=start : 
                    find_face = solid.Shape.Faces[i]
                    u,v = find_face.Surface.parameter(find_face.CenterOfMass)
                    nrml = find_face.normalAt(u,v)
                    if nrml.dot(vec) >= 0.75:
                        ref_list.append((solid, f"Face{i+1}"))
                        ref_indx.append(i) 

        print(f'num of faces as force constraints: {len(ref_list)}')
        self.doc.FemConstraintForce.References  = ref_list
        #set
        self.doc.FemConstraintForce.Direction = (self.doc.RefBox,["Face6"])
        self.doc.FemConstraintForce.Reversed = True
        self.doc.FemConstraintForce.Scale = 1
        self.doc.recompute()
        
        return self.doc, ref_indx

    def add_constraints_to_scene_meshes(self,constraint_indx,color,view_markers=True):
        
        _color = [255,0,0,0] if color=='red' else [0,0,255,0]
        
        N = len(constraint_indx)
        if view_markers:# view the selected faces by marker
            meshes=[]
            for k in range(N):
                center = self.doc.Solid.Shape.Faces[constraint_indx[k]].CenterOfMass        

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
                _ver0 = np.array(self.doc.Solid.Shape.Faces[constraint_indx[k]].Vertexes[0].Point)
                _ver1 = np.array(self.doc.Solid.Shape.Faces[constraint_indx[k]].Vertexes[1].Point)
                _ver2 = np.array(self.doc.Solid.Shape.Faces[constraint_indx[k]].Vertexes[2].Point)

                VEC[ind]   = _ver0
                VEC[ind+1] = _ver1
                VEC[ind+2] = _ver2
            
            constraint_scene = trimesh.Trimesh(vertices=VEC, faces=FAC)
            
        return constraint_scene
            
    def concat_meshes(self,mesh1,mesh2):
        concatenated_mesh = trimesh.util.concatenate(mesh1,mesh2)
        return concatenated_mesh
        
    def create_shape_femmesh(self):
        self.fem_ok = True
        #### femmesh_obj = ObjectsFem.makeMeshNetgen(doc, 'FEMMeshNetgen')#### old function
        femmesh_obj = self.doc.addObject('Fem::FemMeshShapeNetgenObject', 'FEMMeshNetgen')
        femmesh_obj.Fineness = "VeryCoarse"
        femmesh_obj.SecondOrder = False
        femmesh_obj.Optimize = False
        femmesh_obj.Shape = self.doc.Solid # create femmesh from object named:Solid
        try:
            self.doc.recompute()
        except:
            print("GZ")

        if len(femmesh_obj.FemMesh.Faces)>0:
            self.doc.Analysis.addObject(femmesh_obj)
            print("femmesh created from shape")
        else:
            self.fem_ok = False
            print("ERRR - no femmesh is created!")    
        return self.doc, self.fem_ok    
    
    def init_shape(self,load_3d=True): # create state0 - load a 3d mesh or create a cuboid
        self.state0_trimesh = []
        self.mesh_OK = True
        if load_3d:

            self.Mesh_obj_Name = self.loaded_mesh_filename.replace(".obj","")

            Mesh.insert(os.path.join(self.loaded_mesh_path,self.loaded_mesh_filename),self.doc.Name)
            print(f"\n ******* {self.loaded_mesh_filename} is loaded ******* \n")
            
            self.msh = self.doc.getObject(self.Mesh_obj_Name)
            print(self.list_doc_objects())
            self.mesh_OK = self.checkMesh(self.msh.Mesh) # important
            print(f'mesh_OK:{self.mesh_OK}')
            
            
            if self.mesh_OK:
                self.doc.addObject("Part::Feature","initShape")
                s=Part.Shape()
                s.makeShapeFromMesh(self.doc.getObject(self.Mesh_obj_Name).Mesh.Topology,0.100000)
                self.doc.getObject("initShape").Shape = s
                self.doc.getObject("initShape").purgeTouched()
                del s

                sh=self.doc.initShape.Shape
                sh=Part.Solid(sh)
                obj=self.doc.addObject("Part::Feature","Solid")
                obj.Shape=sh
                print(f'shape bounding box: {sh.BoundBox}')
                del sh, obj
                
                self.state0_trimesh = self.Meshobj_to_trimesh(
                    self.doc.getObject(self.Mesh_obj_Name).Mesh)
                self.doc = self.remove_old_shape_result_mesh(name="initShape")
        
        else:        
            Solid_obj = self.doc.addObject("Part::Box", "Solid")
            Solid_obj.Height = 2
            Solid_obj.Width = 5
            Solid_obj.Length = 100
            
            self.Mesh_obj_Name = "initBox"

            self.msh = self.doc.addObject("Mesh::Feature",self.Mesh_obj_Name)
            prt = self.doc.getObject("Solid")
            shp = prt.Shape.copy(False)
            shp.Placement = prt.getGlobalPlacement()
            self.msh.Mesh=MeshPart.meshFromShape(
                Shape=shp,
                Fineness=2,
                SecondOrder=0,
                Optimize=1,
                AllowQuad=0)
            del prt, shp

            self.state0_trimesh = self.Meshobj_to_trimesh(self.doc.getObject(self.Mesh_obj_Name).Mesh)
            self.doc = self.remove_old_shape_result_mesh(name=self.Mesh_obj_Name)
        
        self.doc = self.remove_old_mesh_result_mesh(name=self.Mesh_obj_Name)
        return self.doc, self.state0_trimesh, self.Mesh_obj_Name, self.mesh_OK

    def checkMesh(self,msh): # IMPORTANT : msh type is msh.Mesh!
        mesh_OK = True
        invalid_points = Mesh.Mesh.hasInvalidPoints(msh)
        has_manifolds = Mesh.Mesh.hasNonManifolds(msh)
        orient_faces = Mesh.Mesh.hasNonUniformOrientedFacets(msh)
        self_intersect = Mesh.Mesh.hasSelfIntersections(msh)
        
        if (invalid_points) or (has_manifolds) or (orient_faces) or (self_intersect):
            mesh_OK = False
            print("--------")
            print(f"invalid_points: {invalid_points}, has_manifolds: {has_manifolds}, orient_faces: {orient_faces}, self_intersect: {self_intersect}")
            print("--------")
            print("     XXX Mesh is not healthy!  XXX ")
        else:
            print("     +++ Mesh is healthy!  +++ ")    

        return mesh_OK


    def Meshobj_to_trimesh(self,mesh_obj):
        v,f = mesh_obj.Topology
        mytrimesh = trimesh.Trimesh(vertices=v,faces=f)
        return mytrimesh
    
    def remove_old_femmesh(self):
        femmesh_list = self.doc.findObjects("Fem::FemMeshShapeNetgenObject")
        if femmesh_list:
            self.doc.removeObject(femmesh_list[0].Name)
            self.doc.recompute()
            print("old femmesh is removed...")
        return self.doc

    def remove_old_shape_result_mesh(self,name="Mesh001"):
        shape_result_mesh = self.doc.getObject(name)
        if shape_result_mesh:
            self.doc.removeObject(shape_result_mesh.Name)
            print("shape result mesh is removed...")
        return self.doc

    def remove_old_mesh_result_mesh(self,name="Mesh"):
        mesh_result_mesh = self.doc.getObject(name)
        if mesh_result_mesh:
            self.doc.removeObject(mesh_result_mesh.Name)
            print("mesh result mesh is removed...")
        return self.doc 

    def list_doc_objects(self):
        print('---------------------------')
        for obj_ in self.doc.Objects:
            print(f'{obj_.Name},    --> {type(obj_)}')
        print('---------------------------')

    def document_setup(self):
        self.doc = self.add_ref_object() 
        self.doc = self.create_analysis()
        self.doc = self.add_solver()
        self.doc = self.add_material()

        self.doc = self.create_constraint_fixed()
        self.doc = self.create_constraint_force()
        return self.doc

    
    def add_ref_object(self): 
        # ref object is needed to keep the constraints dependencies (for clearning the doc)
        # and use this refBox for normal directions (initialize the constraints)
        box_obj = self.doc.addObject('Part::Box', 'RefBox')
        box_obj.Height = 1
        box_obj.Width = 1
        box_obj.Length = 1
        return self.doc    

    def create_analysis(self):
        analysis_object = ObjectsFem.makeAnalysis(self.doc, 'Analysis')
        return self.doc
    
    def add_solver(self):
        self.doc.Analysis.addObject(ObjectsFem.makeSolverCalculixCcxTools(self.doc))
        return self.doc

    
    def add_material(self,selected_material='steel'):
        PET = {'Name': 'PET', 'YoungsModulus': '3150 MPa' ,'PoissonRatio': '0.36', 'Density': '1380 kg/m^3'}
        STEEL = {'Name' : 'Steel-Generic','YoungsModulus': '210000 MPa','PoissonRatio': '0.30','Density': '7900 kg/m^3'}

        material_object = ObjectsFem.makeMaterialSolid(self.doc, "SolidMaterial")
        mat = material_object.Material
        
        if selected_material=='PET':
            mat['Name'] = PET['Name']
            mat['YoungsModulus'] = PET['YoungsModulus']
            mat['PoissonRatio'] = PET['PoissonRatio']
            mat['Density'] = PET['Density']    
        else:    
            mat['Name'] = STEEL['Name']
            mat['YoungsModulus'] = STEEL['YoungsModulus']
            mat['PoissonRatio'] = STEEL['PoissonRatio']
            mat['Density'] = STEEL['Density']
        
                
        material_object.Material = mat
        self.doc.Analysis.addObject(material_object)
        return self.doc

    def create_constraint_fixed(self):
        self.doc.addObject("Fem::ConstraintFixed","FemConstraintFixed")
        self.doc.FemConstraintFixed.Scale = 1
        self.doc.FemConstraintFixed.References = [(self.doc.RefBox,"Face2"), (self.doc.RefBox,"Face1")]
        self.doc.Analysis.addObject(self.doc.FemConstraintFixed)
        self.doc.recompute()
        return self.doc

    def create_constraint_force(self):
        self.doc.addObject("Fem::ConstraintForce","FemConstraintForce")
        self.doc.FemConstraintForce.Force = 8000
        self.doc.FemConstraintForce.Direction = (self.doc.RefBox,["Face6"])# **** direction Z axis
        self.doc.FemConstraintForce.Reversed = False
        self.doc.FemConstraintForce.Scale = 1
        self.doc.FemConstraintForce.References = [(self.doc.RefBox,"Face6")]
        self.doc.Analysis.addObject(self.doc.FemConstraintForce)
        self.doc.recompute()
        return self.doc

    def save_state(self):
        now = str(datetime.datetime.now())
        sav = "_"+now.replace("-","").replace(" ","").replace(":","").replace(".","")
        
        state_info = f"{self.initMesh_Name}_{self.force_position}_{self.force_val}"
        print(f"state_info : {state_info}")
        saved_filename = f"{sav}.obj"
        self.save_trimesh(self.result_trimesh,self.save_path,saved_filename)
        
        row_data = [f"{self.initMesh_Name}.obj",self.force_position,self.force_val,saved_filename]
        
        self.add_to_excel_file(row_data,xls_file=self.xls_filename,xls_pth=self.xls_pth)

    
    def add_to_excel_file(self,row_data,xls_file,xls_pth):

        xls_obj = os.path.join(xls_pth,xls_file)
        
        if(row_data):
            if not os.path.exists(xls_obj):
                wbook = Workbook()
            else:
                wbook = openpyxl.load_workbook(xls_obj)
                
            wsheet = wbook.worksheets[0]
            wsheet.append(row_data)    
            wbook.save(xls_obj)

    def save_trimesh(self,mytrimesh, pth ,saved_filename):
        file_obj = os.path.join(pth,saved_filename)
        mytrimesh.export(file_obj)
        print(f'file is saved: {file_obj}')
    
    def generate_complete_meshes_scene(self,trimesh_scene_meshes):
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
    
    def create_shape_femmesh_gmsh(self): # femmesh using gmsh - not integrated yet # To Do 
        femmesh_obj = ObjectsFem.makeMeshGmsh(self.doc, "GMSH"+"_202105_initBox")
        femmesh_obj.Part = self.doc.Solid
        self.doc.recompute()
        gmsh_mesh = gt(femmesh_obj)
        error = gmsh_mesh.create_mesh()
        print(error)

        self.doc.Analysis.addObject(femmesh_obj)
        
    def reset_env(self):
        self.doc = self.remove_old_femmesh()
        self.doc = self.clear_constraints()
        self.doc = self.remove_old_solid()
        
        keep_objs=['RefBox',
                   'Analysis',
                   'SolidMaterial',
                   'FemConstraintForce',
                   'FemConstraintFixed',
                   'CalculiXccxTools']
        
        for objct in self.doc.Objects:
            if not objct.Name in keep_objs:
                print(f'removed: {objct.Name}')
                self.doc.removeObject(objct.Name)
        
        del self.trimesh_scene_meshes
        self.trimesh_scene_meshes = []
        del self.constraint_scene_meshes
        self.constraint_scene_meshes = []
        
        self.fail = False
        self.doc,self.state0_trimesh, self.initMesh_Name, self.mesh_OK = self.init_shape(self.load_3d)
        if not self.mesh_OK: print('*** init mesh is not acceptable!')
        self.trimesh_scene_meshes.append(self.state0_trimesh)
        self.step_no = 0
        return self.state0_trimesh
    
    def view_all_states(self):
        i=0
        final_scene = []
        for mm in self.trimesh_scene_meshes:
            center = [i*11, 0, 0]
            mm.apply_translation(center)
            final_scene.append(mm) 
            i+=1
        return final_scene
