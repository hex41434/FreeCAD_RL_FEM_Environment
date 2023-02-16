import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
import sys
# import pymeshlab
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
import datetime as dt
import random
import openpyxl
from openpyxl import Workbook
import yaml
from scipy.spatial import cKDTree as KDTree
from termcolor import colored
from PIL import Image
import fileinput
class FreeCADWrapper(object):
    """Class allowing to interact with the FreeCad simulation.

    TODO: more description
    """
    # def __init__(self, **kwargs):
    def __init__(self, render_modes,
                 save_path,
                 loaded_mesh_path,
                 loaded_mesh_filename,
                 gt_mesh_path,
                 view_meshes,
                 xls_pth,
                 xls_filename,
                 load_3d):
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
            'xls_filename',
            'render_mode',
            'load_3d'}

        self.save_path = cfg['save_path']
        self.loaded_mesh_path = cfg['loaded_mesh_path']
        self.loaded_mesh_filename = cfg['loaded_mesh_filename']
        self.gt_mesh_path = cfg['gt_mesh_path']
        self.view_meshes = cfg['view_meshes']
        self.xls_pth = cfg['xls_pth']
        self.xls_filename = cfg['xls_filename']
        self.render_mode = cfg['render_mode']
        self.load_3d = cfg['load_3d']

        # self.__dict__.update((k, v) for k, v in kwargs.items() if (k in allowed_keys) and not (v is None)) TODO
        
        # for visualizing the fem constraints on the mesh
        self.trimesh_scene_meshes = []
        self.constraint_scene_meshes = []
        self.view_meshes = view_meshes #True # for jupyter nb TODO
        
        self.inp_dir = os.path.join(os.getcwd(),'my_inp_folder')
        self.inpfile = 'FEMMeshNetgen.inp'
         
        self.doc, self.state0_trimesh,self.mesh_OK = self.document_setup()
        
        if not self.mesh_OK: print('*** init mesh is not acceptable!') 
        self.trimesh_scene_meshes.append(self.state0_trimesh)
        
        self.im_observation = []#np.asarray(Image.open('/scratch/aifa/MyRepo/RL_FEM/gym_RLFEM/FreeCAD_RL_FEM_Environment/scripts/data/Fcad22_ver1.png')) # TODO        

    def create_fem_analysis(self,action):
        self.action = action
        #clear last ccxResults to run next fea
        self.fea.purge_results()
        #write inp file 
        self.write_updated_inp_for_fea()
        # 
        # TODO needed?
        # self.fixed_scene = []#self.add_constraints_to_scene_meshes(self.fixed_indx,color='blue')
        # self.force_scene = []#self.add_constraints_to_scene_meshes(self.force_indx,color='red')
        # self.constraint_scene_meshes.append(self.concat_meshes(self.fixed_scene,self.force_scene))

    def run_ccx_updated_inp(self):
        cc = self.fea.ccx_run()
        print(cc)
        try:
            self.fea.load_results()
        except:
            self.fem_ok = False
        return self.doc, self.fea ,self.fem_ok
    
    def write_updated_inp_for_fea(self):
        self.prepare_inp_node_block()
        self.update_inp_file('Nodes')
        self.prepare_inp_cload_block()
        self.update_inp_file(BLOCK='CLOAD')
        
    def prepare_inp_node_block(self):
        with open(os.path.join(self.inp_dir,"Nodes_Volumes.txt"), "w") as text_file_nodes:
            ccx_float_format = "{:.13E}".format    
            for nd in self.new_Nodes:
                write_str = f'{nd[0]}, {ccx_float_format(nd[1])},{ccx_float_format(nd[2])},{ccx_float_format(nd[3])}\n'
                text_file_nodes.write(write_str)
            
            text_file_nodes.write('\n\n') 
    
    # prepare force constraint for inp file format
    def prepare_inp_cload_block(self):
        Fx, Fy, frc = self.action
        frc = "{:.13E}".format(frc)
        region = 0
        print(frc,Fx,Fy)
        with open(os.path.join(self.inp_dir,"ConstForce.txt"), "w") as text_file_force:
            for it in self.doc.FEMMeshNetgen.FemMesh.Nodes.items():
                if np.ceil(it[1].x) == Fx and np.ceil(it[1].y) == Fy:
                # if np.ceil(it[1].x) >= Fx-region and np.ceil(it[1].x) <= Fx+region and np.ceil(it[1].y) >= Fy-region and np.ceil(it[1].y) <= Fy+region:
                    write_str = f'{it[0]},3,{frc}\n'
                    text_file_force.write(write_str)
            
            text_file_force.write('\n\n***********************************************************\n')#end of block  
    
    def update_inp_file(self,BLOCK):
        if BLOCK=='Nodes':
            #NODES
            HEAD = '*Node, NSET=Nall'
            # TAIL = '** Define element set Eall'
            TAIL ='** Volume elements'
            
        elif BLOCK=='CLOAD':

            #CLOAD
            HEAD = '** node loads on shape:'
            TAIL = '** Outputs --> frd file'
            
        self.create_head_tail_files(HEAD,TAIL,BLOCK)    
        self.insert_inp_blocks(HEAD,TAIL,BLOCK) 
    
    def insert_inp_blocks(self,HEAD,TAIL,BLOCK):              
        inp_dir = self.inp_dir
        inpfile = self.inpfile
        cwd = os.getcwd()
        os.chdir(inp_dir)
            
        if BLOCK=='Nodes':
            block_file = 'Nodes_Volumes.txt' 
            file_list = ['head_nodes.txt', block_file, 'tail_nodes.txt']
        elif BLOCK=='CLOAD':
            block_file = 'ConstForce.txt'   
            file_list = ['head_cload.txt', block_file, 'tail_cload.txt']
            
        
        with open(os.path.join(inp_dir,inpfile), 'w') as file:
            input_lines = fileinput.input(file_list)
            file.writelines(input_lines)
        os.chdir(cwd)
        
    def create_head_tail_files(self,HEAD,TAIL,BLOCK): # HEAD:included, TAIL:excluded
        inp_dir = self.inp_dir
        inpfile = self.inpfile
        
        ln = 0
        start_line = 0
        end_line = 0
        if BLOCK=='Nodes':
            with open(os.path.join(inp_dir,inpfile),"r") as fin, open(os.path.join(inp_dir,'head_nodes.txt'),"w") as fout_head, open(os.path.join(inp_dir,'tail_nodes.txt'),"w") as fout_tail:
                for line in fin:
                    ln = ln+1
                    if start_line==0:
                        fout_head.write(line)
                        if HEAD in line:
                            start_line = ln
                    
                    if TAIL in line: #TAIL is excluded
                            end_line = ln-1
                    
                    if end_line!=0:
                        fout_tail.write(line)
        
        if BLOCK=='CLOAD':
            with open(os.path.join(inp_dir,inpfile),"r") as fin, open(os.path.join(inp_dir,'head_cload.txt'),"w") as fout_head, open(os.path.join(inp_dir,'tail_cload.txt'),"w") as fout_tail:
                for line in fin:
                    ln = ln+1
                    if start_line==0:
                        fout_head.write(line)
                        if HEAD in line:
                            start_line = ln
                    
                    if TAIL in line: #TAIL is excluded
                            end_line = ln-1
                    
                    if end_line!=0:
                        fout_tail.write(line)
    
    def reset_femmesh(self):
        new_Nodes = []
        
        res_nd_no = self.res_nd_no_0
        res_mesh = self.res_mesh_0 
        for n in res_nd_no: #add Nodes
            nx = (res_mesh.Nodes[n].x)
            ny = (res_mesh.Nodes[n].y)
            nz = (res_mesh.Nodes[n].z)
            #node ids starts from 1 
            new_Nodes.append((n,nx,ny,nz))
        return new_Nodes
                
    def compute_node_displacements(self):
        new_Nodes = []
        res_coord = self.doc.CCX_Results.DisplacementVectors
        res_nd_no = self.doc.CCX_Results.NodeNumbers
        res_mesh = self.doc.ResultMesh.FemMesh
        for n in res_nd_no: #add Nodes
            nx = (res_mesh.Nodes[n].x + res_coord[n-1].x)
            ny = (res_mesh.Nodes[n].y + res_coord[n-1].y)
            nz = (res_mesh.Nodes[n].z + res_coord[n-1].z)
            #node ids starts from 1 
            new_Nodes.append((n,nx,ny,nz))
        
        return new_Nodes        

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
        femmesh_obj.Fineness = "Fine"
        femmesh_obj.SecondOrder = False
        femmesh_obj.Optimize = False
        femmesh_obj.MaxSize = .5
        # create femmesh from object named:Solid
        femmesh_obj.Shape = self.doc.Solid 
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
  
    def Meshobj_to_trimesh(self,out_mesh):
        v,f = out_mesh.Topology
        mytrimesh = trimesh.Trimesh(vertices=v,faces=f)
        return mytrimesh

    def resultmesh_to_trimesh(self):
        femmesh_obj = self.doc.getObject('ResultMesh').FemMesh
        result = self.doc.getObject('CCX_Results')
        out = femmesh.femmesh2mesh.femmesh_2_mesh(femmesh_obj, result)
        out_mesh = Mesh.Mesh(out)
        trmsh = self.Meshobj_to_trimesh(out_mesh)
        del out
        
        return trmsh, out_mesh

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
            print(colored("     XXX Mesh is not healthy!  XXX ",'red'))    

        return mesh_OK

    def list_doc_objects(self):
        print('---------------------------')
        for obj_ in self.doc.Objects:
            print(f'{obj_.Name},    --> {type(obj_)}')
        print('---------------------------')

    def document_setup(self): 
        self.doc = FreeCAD.newDocument('doc')
        self.doc = self.create_analysis()
        self.doc = self.add_solver()
        self.doc = self.add_material()
        
        self.fea = ccxtools.FemToolsCcx()
        
        # to create mesh from our initial state, we need a fem step with a very small force
        self.state0_trimesh = []
        self.mesh_OK = True
        self.fem_ok = True
        self.step_no = 0
        self.action = (0,0,0)
        
        self.load_3d=False # TODO?!
        if(not self.load_3d):
            Solid_obj = self.doc.addObject("Part::Box", "Solid")
            Solid_obj.Height = 2
            Solid_obj.Width = 3
            Solid_obj.Length = 10
        
        self.doc, self.fem_ok = self.create_shape_femmesh()
        
        self.doc = self.create_constraint_fixed()
        self.doc = self.create_constraint_force()
        
        self.list_doc_objects() 
        
        fea1=self.fea
        fea1.update_objects()
        fea1.setup_working_dir(self.inp_dir) 
        
        fea1.setup_ccx()
        message = fea1.check_prerequisites()
        if not message:
            fea1.purge_results()
            fea1.write_inp_file()
            cc = fea1.ccx_run()
            print(f'cc:{cc}')
            print(f'+o+o+o+o+o+o+ .inp filename {fea1.inp_file_name}')
            fea1.load_results()
        else:
            print("Oh, we have a problem! {}\n".format(message))  # in python console

        # i need to keep the state0 and the initial femmesh (femmesh0) --> (for reset function)
        self.state0_trimesh,self.result_FCmesh = self.resultmesh_to_trimesh() 
        self.result_trimesh = self.state0_trimesh
            
        self.res_nd_no_0 = self.doc.CCX_Results.NodeNumbers
        self.res_mesh_0 = self.doc.ResultMesh.FemMesh
        self.old_state_name = 'Mesh0.obj'
        
        # self.save_state() -> only needed once to write Mesh0.obj file 
        
        # we need to keep the mesh fixed -> manually update the node coordinates in inp file 
        self.new_Nodes = self.reset_femmesh()
        
        return self.doc ,self.state0_trimesh, self.mesh_OK # 2 last objs needed?!

    def create_analysis(self):
        analysis_object = ObjectsFem.makeAnalysis(self.doc, 'Analysis')
        return self.doc
    
    def add_solver(self):
        self.doc.Analysis.addObject(ObjectsFem.makeSolverCalculixCcxTools(self.doc))
        return self.doc

    def add_material(self,selected_material='PET'):
        PET = {'Name': 'PET', 'YoungsModulus': '3150 MPa' ,'PoissonRatio': '0.36', 'Density': '1380 kg/m^3'}
        STEEL = {'Name' : 'Steel-Generic','YoungsModulus': '210000 MPa','PoissonRatio': '0.30','Density': '7900 kg/m^3'}

        material_object = ObjectsFem.makeMaterialSolid(self.doc, "SolidMaterial")
        mat = material_object.Material
        
        if selected_material=='PET':
            mat['Name'] = PET['Name']
            mat['YoungsModulus'] = PET['YoungsModulus']
            mat['PoissonRatio'] = PET['PoissonRatio']
            mat['Density'] = PET['Density']    
        elif selected_material=='steel':    
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
        self.doc.FemConstraintFixed.References = [(self.doc.Solid,"Face5")] #bottom surface
        self.doc.Analysis.addObject(self.doc.FemConstraintFixed)
        self.doc.recompute()
        return self.doc

    def create_constraint_force(self):
        self.doc.addObject("Fem::ConstraintForce","FemConstraintForce")
        self.doc.FemConstraintForce.Force = 1
        self.doc.FemConstraintForce.Direction = (self.doc.Solid,["Face6"])# **** direction Z axis
        self.doc.FemConstraintForce.Reversed = False
        self.doc.FemConstraintForce.Scale = 1
        self.doc.FemConstraintForce.References = [(self.doc.Solid,"Face6")]
        self.doc.Analysis.addObject(self.doc.FemConstraintForce)
        self.doc.recompute()
        return self.doc

    def save_state(self):
        timestamp_ = "_"+dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_mesh_filename = f"{timestamp_}_step_{self.step_no}.obj"
        # self.save_trimesh(self.result_trimesh,self.save_path,f"{timestamp_}_trimesh_step_{self.step_no}.obj")
        self.save_mesh_obj_from_nodes(self.save_path,saved_mesh_filename)
        # self.im_observation = self.mesh_to_observation(timestamp_)

        Fx = self.action[0]
        Fy = self.action[1]
        F = self.action[2]
        row_data = [self.old_state_name,Fx,Fy,F,saved_mesh_filename]
        self.old_state_name = saved_mesh_filename
        self.add_to_excel_file(row_data,xls_file=self.xls_filename,xls_pth=self.xls_pth)
    
    def save_mesh_obj_from_nodes(self,save_path,saved_mesh_filename):
        vertex_coordinates = ''

        with open(os.path.join(save_path,saved_mesh_filename), "w") as f:    
            for n in self.node_vertex_map_list():
                vertex_coordinates = f'v {self.new_Nodes[n-1][1]} {self.new_Nodes[n-1][2]} {self.new_Nodes[n-1][3]}\n'# Node index from 1
                f.write(vertex_coordinates)
            
            with open('/scratch/aifa/MyRepo/RL_FEM/gym_RLFEM/FreeCAD_RL_FEM_Environment/data/Mesh0_template_face_indx.txt') as readfile:
                line = readfile.readline()
                f.write(line)
                while line:
                   line = readfile.readline()
                   f.write(line)
                   
         
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
    
    def create_shape_femmesh_gmsh(self): # femmesh using gmsh - not integrated yet # TODO
        femmesh_obj = ObjectsFem.makeMeshGmsh(self.doc, "GMSH"+"_202105_initBox")
        femmesh_obj.Part = self.doc.Solid
        self.doc.recompute()
        gmsh_mesh = gt(femmesh_obj)
        error = gmsh_mesh.create_mesh()
        print(error)

        self.doc.Analysis.addObject(femmesh_obj)
        
    def reset_env(self):
        
        del self.trimesh_scene_meshes
        self.trimesh_scene_meshes = []
        del self.constraint_scene_meshes
        self.constraint_scene_meshes = []
        
        self.fem_ok = True
        self.result_trimesh = self.state0_trimesh
        self.old_state_name = 'Mesh0.obj'
        self.trimesh_scene_meshes.append(self.result_trimesh)
        self.step_no = 0
        print(f'reset_func --> step no:{self.step_no}') 
        self.new_Nodes = self.reset_femmesh()
        return self.result_trimesh

    
    def view_all_states(self):
        i=0
        final_scene = []
        for mm in self.trimesh_scene_meshes:
            center = [i*11, 0, 0]
            mm.apply_translation(center)
            final_scene.append(mm) 
            i+=1
        return final_scene

    # Copyright 2004-present Facebook. All Rights Reserved.
    def compute_trimesh_chamfer(gt_points, gen_mesh, offset=0, scale=1, num_mesh_samples=30000):
        """
        This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

        gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
                compute_metrics.ply for more documentation)

        gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
                method (see compute_metrics.py for more)

        """
        gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]
        gen_points_sampled = gen_points_sampled / scale - offset

        # only need numpy array of points
        gt_points_np = gt_points.vertices

        # one direction
        gen_points_kd_tree = KDTree(gen_points_sampled)
        one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
        gt_to_gen_chamfer = np.mean(np.square(one_distances))

        # other direction
        gt_points_kd_tree = KDTree(gt_points_np)
        two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
        gen_to_gt_chamfer = np.mean(np.square(two_distances))
        print(gt_to_gen_chamfer + gen_to_gt_chamfer)
        return gt_to_gen_chamfer + gen_to_gt_chamfer

    def mesh_to_observation(self,timestamp_):
        colored_result_mesh = self.trimesh_to_color_depth(self.result_trimesh)
        rgbimg, depth_img = self.make_observation(colored_result_mesh)
        self.im_observation = Image.fromarray(rgbimg).resize([64,48]).crop(box=[5,15,60,38])
        save_obs=False
        if(save_obs):
            self.im_observation.save(os.path.join(self.save_path,f'{timestamp_}.png'))
        # self.im_observation = np.asarray(self.im_observation) # img or array
        return self.im_observation
    
    def make_observation(self,trmsh):
        camera_pres = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        s = np.sqrt(2)/2
        camera_pres_pose_top = np.array([
        [s, 0.0, s, 5.0],
        [0.0, 1.0, 0.0, 2.0],
        [s, 0.0, s, 10],
        [0.0, 0.0, 0.0, 1.0],
        ])
                
        mish = pyrender.Mesh.from_trimesh(trmsh)
        prscene = pyrender.Scene(ambient_light=np.array([1.0, 1.0, 1.0, 1.0]))
        prscene.add(camera_pres,pose=camera_pres_pose_top)
        prscene.add(mish)
        
        r = pyrender.OffscreenRenderer(viewport_width=640,viewport_height=480,point_size=1.0)
        img,dpth = r.render(prscene)
        print(f"============={img.size}")
        return img, dpth
    
    def trimesh_to_color_depth(self,trmsh):
        kaf = np.array(trmsh.vertices)
        kaf[:,2]=0

        radii = np.linalg.norm(trmsh.vertices - kaf, axis=1)
        cmap = ['jet' ,'viridis']
        trmsh.visual.vertex_colors = trimesh.visual.interpolate(radii, color_map=cmap[0])
        return trmsh
    
    def node_vertex_map_list(self):
        map = [7,
            5,
            57,
            58,
            59,
            60,
            61,
            101,
            77,
            316,
            99,
            75,
            314,
            97,
            73,
            312,
            95,
            71,
            23,
            459,
            310,
            193,
            398,
            47,
            175,
            93,
            69,
            308,
            91,
            67,
            306,
            123,
            89,
            65,
            304,
            3,
            1,
            81,
            82,
            83,
            84,
            85,
            87,
            63,
            103,
            79,
            427,
            366,
            383,
            444,
            28,
            52,
            214,
            466,
            405,
            228,
            158,
            369,
            430,
            488,
            489,
            31,
            55,
            19,
            189,
            43,
            470,
            392,
            15,
            232,
            167,
            114,
            134,
            241,
            409,
            493,
            422,
            361,
            380,
            222,
            152,
            441,
            166,
            496,
            137,
            227,
            157,
            39,
            495,
            323,
            247,
            319,
            245,
            321,
            243,
            407,
            468,
            492,
            218,
            376,
            437,
            412,
            494,
            126,
            165,
            415,
            354,
            143,
            462,
            401,
            176,
            122,
            210,
            428,
            367,
            384,
            445,
            29,
            199,
            53,
            429,
            368,
            283,
            480,
            479,
            481,
            467,
            406,
            159,
            108,
            257,
            168,
            148,
            135,
            396,
            207,
            318,
            425,
            364,
            138,
            115,
            198,
            300,
            296,
            294,
            292,
            290,
            20,
            190,
            44,
            172,
            109,
            267,
            337,
            265,
            335,
            263,
            282,
            281,
            277,
            347,
            275,
            345,
            273,
            343,
            420,
            271,
            359,
            341,
            269,
            357,
            339,
            177,
            211,
            385,
            446,
            180,
            200,
            30,
            54,
            451,
            390,
            431,
            370,
            485,
            472,
            487,
            486,
            484,
            483,
            38,
            14,
            164,
            182,
            131,
            501,
            219,
            377,
            438,
            136,
            24,
            460,
            194,
            399,
            48,
            127,
            286,
            288,
            450,
            454,
            393,
            413,
            389,
            121,
            21,
            457,
            223,
            381,
            153,
            442,
            178,
            386,
            447,
            231,
            478,
            471,
            169,
            149,
            352,
            298,
            141,
            142,
            161,
            140,
            234,
            418,
            463,
            402,
            195,
            49,
            285,
            162,
            163,
            120,
            10,
            13,
            12,
            119,
            117,
            349,
            11,
            181,
            9,
            328,
            252,
            326,
            250,
            324,
            248,
            334,
            330,
            254,
            201,
            6,
            100,
            76,
            315,
            98,
            74,
            313,
            96,
            72,
            311,
            94,
            70,
            309,
            92,
            68,
            307,
            90,
            66,
            305,
            118,
            88,
            64,
            303,
            104,
            80,
            86,
            62,
            102,
            78,
            317,
            208,
            426,
            365,
            8,
            179,
            448,
            387,
            261,
            280,
            259,
            260,
            348,
            416,
            355,
            37,
            4,
            35,
            2,
            36,
            33,
            34,
            233,
            128,
            191,
            45,
            173,
            394,
            423,
            362,
            421,
            360,
            240,
            212,
            322,
            246,
            320,
            244,
            242,
            132,
            434,
            170,
            502,
            378,
            220,
            439,
            150,
            433,
            372,
            238,
            239,
            236,
            237,
            491,
            25,
            461,
            504,
            505,
            507,
            112,
            106,
            503,
            455,
            373,
            469,
            408,
            258,
            332,
            256,
            203,
            449,
            22,
            458,
            192,
            397,
            46,
            174,
            224,
            382,
            154,
            443,
            497,
            213,
            183,
            432,
            499,
            414,
            353,
            204,
            453,
            371,
            490,
            374,
            435,
            297,
            295,
            293,
            291,
            268,
            338,
            266,
            336,
            264,
            333,
            262,
            278,
            276,
            346,
            274,
            344,
            272,
            342,
            379,
            419,
            270,
            358,
            340,
            464,
            403,
            26,
            196,
            50,
            498,
            482,
            111,
            105,
            410,
            56,
            32,
            202,
            388,
            350,
            500,
            400,
            209,
            411,
            225,
            155,
            235,
            287,
            284,
            289,
            230,
            184,
            217,
            205,
            391,
            129,
            17,
            185,
            41,
            375,
            436,
            187,
            40,
            424,
            363,
            465,
            404,
            27,
            197,
            51,
            145,
            216,
            18,
            188,
            42,
            452,
            113,
            475,
            473,
            476,
            107,
            474,
            477,
            186,
            299,
            16,
            133,
            146,
            171,
            417,
            356,
            221,
            151,
            440,
            124,
            226,
            156,
            139,
            302,
            279,
            301,
            215,
            116,
            229,
            147,
            110,
            456,
            395,
            206,
            160,
            508,
            125,
            506,
            329,
            253,
            327,
            251,
            325,
            130,
            249,
            331,
            255,
            351,
            144]

        return map