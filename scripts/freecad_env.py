import os
import sys

sys.path.append(os.path.join(os.environ['CONDA_PREFIX'], 'lib'))
import FreeCAD
import FreeCADGui
import Part
import ObjectsFem
# from femguiobjects import FemGui
# import FemGui
# from Mod import Fem
import Fem
import Mesh
import MeshPart
import femmesh.femmesh2mesh
from femtools import ccxtools
import numpy as np
import yaml
import pickle
import utils_3d


class freecad_env():
    def __init__(self, **kwargs):
        env_config_path = './env_config.yaml'
        with open(env_config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        allowed_keys = {
            'path',
            'force_value',
            'force_factor',
            'num_actions',
            'flag_rand_action',
            'flag_save',
            'action_type',
            'max_triangles'
        }

        self.path = cfg['path']['load_path']
        self.force_value = cfg['force_params']['force_value']
        self.force_factor = cfg['force_params']['force_factor']
        self.num_actions = cfg['num_actions']
        self.flag_rand_action = cfg['flags']['flag_rand_action']
        self.flag_save = cfg['flags']['flag_save']
        self.action_type = cfg['flags']['action_type']  # 'circle' or 'rectangle'
        self.max_triangles = cfg['max_triangles']

        self.__dict__.update((k, v) for k, v in kwargs.items() if (k in allowed_keys) and not (v is None))

        # TODO: add ''' import FreeCADGui FreeCADGui.showMainWindow() ''' in case of visualization
        # TODO: add the number of actions (sequence) and repeat until it's satisfied
        # self.doc = FreeCAD.getDocument('Unnamed') # CHANGE LATER!
        self.doc = FreeCAD.newDocument('doc')

        self.initialize_doc()

    def clear_doc(self):
        for obj in self.doc.Objects:
                self.doc.removeObject(obj.Name)

    def initialize_doc(self):
        self.flag_first_run = 1
        self.count_action = 0

        self.flag_success = 1
        self.flag_break = 0

        self.region_values_vec = []
        self.force_dir_str_vec = []

        if not (self.path == ''):
            self.insert_mesh()
        else:
            # TODO: correct the dimensions
            self.create_initial_cube()

        self.initialize_obj()
        self.calc_obj_info()
        self.box_obj = self.doc.addObject('Part::Box', 'Box')
        # box_obj.ViewObject.Visibility = False

    def initialize_obj(self):
        if (self.flag_first_run == 1):
            self.obj = self.doc.Objects[-1]
        else:
            self.obj = self.doc.getObject('Mesh_{:02d}'.format(self.count_action - 1))

        if (self.obj.Module == 'Mesh'):
            self.convert2solidmesh()
            self.obj = self.doc.Objects[-1]
            # if (self.flag_first_run == 0):
            #     self.doc.removeObject(self.doc.getObject('Mesh'))
    def calc_obj_info(self):
        self.v_CoM, self.v_NoF = self.extract_info_mesh(self.obj)

        self.min_vals = self.v_CoM.min(axis=0)
        self.max_vals = self.v_CoM.max(axis=0)
        self.obj_dims = abs(self.max_vals - self.min_vals)
        self.LENGTH = self.obj_dims[0]  # x direction
        self.WIDTH = self.obj_dims[1]  # y direction
        self.HEIGHT = self.obj_dims[2]  # z direction

        # TODO: arbitrary numbers, give as parameters
        self.NUM_BIN = 1000
        self.NUM_CS = 20 # number of cross-sections, in case of rectangular actions space

        self.flag_elong = 'length'
        self.calc_epsilon()

        self.v_CoM_sorted = self.get_sorted_CoM(self.v_CoM, self.column_index)

    def calc_epsilon(self):
        if (self.flag_elong == 'length'):
            self.EPSILON = self.LENGTH / self.NUM_BIN
            self.column_index = 0
        elif (self.flag_elong == 'width'):
            self.EPSILON = self.WIDTH / self.NUM_BIN
            self.column_index = 1
        elif (self.flag_elong == 'height'):
            self.EPSILON = self.HEIGHT / self.NUM_BIN
            self.column_index = 2


    def insert_mesh(self):
        Mesh.insert(self.path)

    def create_initial_cube(self):
        obj = self.doc.addObject('Part::Box', 'Box')
        obj.Height = 2
        obj.Width = 5
        obj.Length = 50
        msh = self.doc.addObject('Mesh::Feature', 'Cube_Mesh')
        __part__ = obj
        __shape__ = __part__.Shape.copy(False)
        __shape__.Placement = __part__.getGlobalPlacement()
        msh.Mesh = MeshPart.meshFromShape(Shape=__shape__, Fineness=2, SecondOrder=0, Optimize=1, AllowQuad=0)
        msh.Label = 'Cube_Meshed'
        del msh, __part__, __shape__
        self.doc.removeObject(obj.Name)

    def select_visible_obj(self):
        Gui.Selection.clearSelection()
        for obj in FreeCAD.ActiveDocument.Objects:
            if obj.ViewObject.isVisible():
                Gui.Selection.addSelection(obj)
        sel = FreeCADGui.Selection.getSelection()
        return sel

    def covert2solid(self, shape_obj):
        __s__ = shape_obj.Shape
        __s__ = Part.Solid(__s__)
        solid_obj = self.doc.addObject('Part::Feature', shape_obj.Name + '_solid')
        solid_obj.Label = shape_obj.Name + ' (Solid)'
        solid_obj.Shape = __s__
        if (not (self.path == '')) or (self.flag_first_run == 0):
            solid_obj.Shape = solid_obj.Shape.removeSplitter()
        del __s__
        return solid_obj


    def convert2solidmesh(self):
        shape_obj = self.doc.addObject('Part::Feature', self.obj.Name)
        __shape__ = Part.Shape()
        __shape__.makeShapeFromMesh(self.obj.Mesh.Topology, 0.1)
        shape_obj.Shape = __shape__
        shape_obj.purgeTouched()
        solid_obj = self.covert2solid(shape_obj)
        # obj.ViewObject.Visibility = False
        # shape_obj.ViewObject.Visibility = False
        del shape_obj, __shape__, solid_obj


    def extract_info_mesh(self, obj):
        faces = obj.Shape.Faces.copy()
        v_CoM = []
        v_NoF = []
        for i in range(len(faces)):
            face = faces[i]
            v_CoM.append(np.concatenate((np.array(face.CenterOfMass), [i])))
            v_NoF.append(np.array(face.Surface.Axis))
        v_CoM = np.array(v_CoM)
        v_NoF = np.array(v_NoF)
        del faces
        return v_CoM, v_NoF

    def get_sorted_CoM(self, v_CoM, column_index):
        return v_CoM[v_CoM[:, column_index].argsort()]

    def get_faces_constraint_fixed(self):

        mask_1 = (abs(self.v_CoM_sorted[:, self.column_index] - self.min_vals[self.column_index]) < self.EPSILON)
        mask_2 = (abs(self.v_CoM_sorted[:, self.column_index] - self.max_vals[self.column_index]) < self.EPSILON)
        normal_vector = np.zeros(3)
        normal_vector[self.column_index] = 1
        # TODO: 0.95 is arbitrary, it should be given as a parameter.
        mask_normal_1 = np.dot(self.v_NoF[self.v_CoM_sorted[mask_1, -1].astype(int), :], -normal_vector) > 0.95
        mask_normal_2 = np.dot(self.v_NoF[self.v_CoM_sorted[mask_2, -1].astype(int), :], normal_vector) > 0.95
        face_fixed_indx = np.concatenate(
            (self.v_CoM_sorted[mask_2, -1].astype(int)[mask_normal_2], self.v_CoM_sorted[mask_1, -1].astype(int)[mask_normal_1]))
        return face_fixed_indx

    def check_valid_region(self, region_values):
        flag_valid = 1
        if (self.action_type == 'circle'):
            c_circle, r_circle = region_values
            mask_c_circle = (abs(self.v_CoM_sorted[:, self.column_index] - c_circle[self.column_index]) < 10 * self.EPSILON)
            if not ((c_circle[self.column_index + 1] >= self.v_CoM_sorted[mask_c_circle, self.column_index + 1].min(axis=0)) & (
                    c_circle[self.column_index + 1] <= self.v_CoM_sorted[mask_c_circle, self.column_index + 1].max(axis=0))):
                print('center of circle does not lie on the mesh')
                # TODO: CHECK radius as well!
                flag_valid = 0

        elif (self.action_type == 'rectangle'):
            coor_rectangle, w_rectangle = region_values
        return flag_valid

    def get_faces_constraint_force(self, region_values, force_dir_str):
        face_indx = []
        if self.action_type == 'circle':
            c_circle, r_circle = region_values
            mask_c_circle_1 = (abs(self.v_CoM_sorted[:, self.column_index] - c_circle[self.column_index]) < r_circle)
            mask_c_circle_2 = np.linalg.norm(self.v_CoM_sorted[mask_c_circle_1, self.column_index:self.column_index + 2] - c_circle,
                                             axis=1) <= r_circle
            face_indx = self.v_CoM_sorted[mask_c_circle_1, -1].astype(int)[mask_c_circle_2]

        elif (self.action_type == 'rectangle'):
            coor_rectangle, w_rectangle = region_values
            mask_rectangle_1 = ((coor_rectangle <= self.v_CoM_sorted[:, self.column_index])
                                & (self.v_CoM_sorted[:, self.column_index] <= coor_rectangle + w_rectangle))
            face_indx = self.v_CoM_sorted[mask_rectangle_1, -1].astype(int)

        force_direction = np.zeros(3)
        if (force_dir_str == 'top'):
            force_direction[self.column_index + 2] = 1
        elif (force_dir_str == 'bottom'):
            force_direction[self.column_index + 2] = -1

        mask_normal = np.dot(self.v_NoF[face_indx, :], force_direction) > 0
        return face_indx[mask_normal]


    def add_FEM_mesh_analysis(self, obj):
        femmesh_obj = self.doc.addObject('Fem::FemMeshShapeNetgenObject', 'FEMMeshNetgen')
        femmesh_obj.Shape = obj
        femmesh_obj.MaxSize = 1
        self.doc.Analysis.addObject(femmesh_obj)


    def add_FEM_constraint_fixed(self, obj):
        self.doc.addObject('Fem::ConstraintFixed', 'FemConstraintFixed')
        self.doc.FemConstraintFixed.Scale = 1
        self.doc.Analysis.addObject(self.doc.FemConstraintFixed)
        self.face_fixed_indx = self.get_faces_constraint_fixed()
        self.doc.FemConstraintFixed.References = list(
            zip([obj] * len(self.face_fixed_indx), ['Face' + str(x + 1) for x in self.face_fixed_indx]))

    def add_FEM_constraint_force(self, obj, box_obj, region_values, force_dir_str):
        self.doc.addObject('Fem::ConstraintForce', 'FemConstraintForce')
        self.doc.Analysis.addObject(self.doc.FemConstraintForce)

        self.face_force_indx = self.get_faces_constraint_force(region_values, force_dir_str)

        self.doc.FemConstraintForce.Force = self.force_value
        self.doc.FemConstraintForce.Direction = (box_obj, ['Face6'])
        self.doc.FemConstraintForce.Reversed = True if force_dir_str == 'top' else False
        self.doc.FemConstraintForce.Scale = self.force_factor # TODO: this doesn't work, find a better way
        self.doc.FemConstraintForce.References = list(
            zip([obj] * len(self.face_force_indx), ['Face' + str(x + 1) for x in self.face_force_indx]))


    def create_FEM_solid(self):
        self.material_object = ObjectsFem.makeMaterialSolid(self.doc, 'SolidMaterial')
        mat = self.material_object.Material
        mat['Name'] = 'Steel-Generic'
        mat['YoungsModulus'] = '210000 MPa'
        mat['PoissonRatio'] = '0.30'
        mat['Density'] = '7900 kg/m^3'
        self.material_object.Material = mat
        # FemGui.getActiveAnalysis().addObject(material_object)
        self.doc.Analysis.addObject(self.material_object)


    def creating_analysis(self):
        self.analysis_object = ObjectsFem.makeAnalysis(self.doc, 'Analysis')
        self.analysis_object.addObject(ObjectsFem.makeSolverCalculixCcxTools(self.doc))

    def analysis_run(self, obj, box_obj, region_values, force_dir_str):
        self.creating_analysis()
        self.add_FEM_mesh_analysis(obj)
        self.add_FEM_constraint_fixed(obj)
        self.add_FEM_constraint_force(obj, box_obj, region_values, force_dir_str)
        self.create_FEM_solid()

    def calc_analysis(self):
        self.flag_success = 0
        fea = ccxtools.FemToolsCcx()
        fea.update_objects()
        fea.setup_working_dir()
        fea.setup_ccx()
        message = fea.check_prerequisites()
        if not message:
            fea.purge_results()
            fea.write_inp_file()
            # on error at inp file writing, the inp file path '' was returned (even if the file was written)
            # if we would write the inp file anyway, we need to again set it manually
            # fea.inp_file_name = '/tmp/FEMWB/FEMMeshGmsh.inp'
            fea.ccx_run()
            fea.load_results()
            self.flag_success = 1
        else:
            FreeCAD.Console.PrintError('Oh, we have a problem! {}\n'.format(message))  # in report view
            print('Oh, we have a problem! {}\n'.format(message))  # in python console

    def get_result(self):
        for m in self.doc.Analysis.Group:
            if m.isDerivedFrom('Fem::FemResultObject'):
                result_object = m
                break
        femmesh_obj = result_object.Mesh.FemMesh

        out_mesh = femmesh.femmesh2mesh.femmesh_2_mesh(femmesh_obj, result_object)
        self.new_mesh = Mesh.Mesh(out_mesh)
        if (len(self.new_mesh.Facets) >= 1.1 * self.max_triangles):
            self.new_mesh = self.mesh_decimation(self.new_mesh)
        Mesh.show(self.new_mesh, 'Mesh_{:02d}'.format(self.count_action))
        # msh = self.doc.getObject('Mesh')
        # msh.Label = 'Mesh'
        # del msh
    def create_mesh(self, mesh):
        vertices = np.array([p.Vector for p in mesh.Points])
        faces = np.array([f.PointIndices for f in mesh.Facets])
        mesh_new = utils_3d.create_mesh_open3d(vertices, faces)
        return mesh_new

    def mesh_decimation(self, mesh):
        mesh_new = self.create_mesh(mesh)
        mesh_new = utils_3d.mesh_decimation_open3d(mesh_new, max_triangles=self.max_triangles)
        return self.create_mesh_obj(mesh_new)

    def create_mesh_obj(self, mesh):
        triangles = utils_3d.get_mesh_info_open3d(mesh)
        return Mesh.Mesh(triangles)

    def save_result_step(self, save_path):
        # TODO: add proper save_path and have num_action included in the name
        ###5
        # export one mesh as .ply file
        __objs__ = []
        __objs__.append(self.doc.getObject('Mesh_{:02d}'.format(self.count_action)))

        save_path = './'

        filename = save_path + 'result_' + str(self.count_action) + '.ply'

        # Mesh.export(__objs__,u'wowply.ply')
        Mesh.export(__objs__, filename)

        del __objs__

    def save_result_all(self, save_path):

        for i in range(self.num_actions):
            obj = [self.doc.getObject('Mesh_{:02d}'.format(i + 1))]
            filename = save_path + 'result_' + str(i + 1) + '.ply'
            Mesh.export(obj, filename)
            del obj


    def save_result_info(self, save_path, pickle_obj):
        with open(save_path + '.pickle', 'wb') as handle:
            pickle.dump(pickle_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def reset_doc_action(self):
        for obj in self.doc.Objects:
            if not (obj.Label.startswith('Mesh_') or obj.Label == 'Box'):
                self.doc.removeObject(obj.Name)
            # elif (obj.Label == 'Mesh'):
            #     obj.Label = 'Mesh_old'

    def run_loop(self, region_values=None, force_dir_str='top'):
        while (self.count_action < self.num_actions):
            self.count_action += 1
            if (self.flag_first_run == 1):
                self.flag_first_run = 0
            else:
                self.reset_doc_action()
                self.initialize_obj()
                self.calc_obj_info()

            if (self.flag_rand_action == 0) and (region_values is None):
                if (self.action_type == 'circle'):
                    # c_circle = np.array([15, 7])
                    c_circle = np.array([40, 7])
                    r_circle = 5
                    force_dir_str = 'top'
                    region_values = (c_circle, r_circle)
                elif (self.action_type == 'rectangle') and (region_values is None):
                    coor_rectangle = np.array([25])
                    w_rectangle = 7
                    force_dir_str = 'top'
                    region_values = (coor_rectangle, w_rectangle)

            if (self.flag_rand_action == 1):
                region_values, force_dir_str = self.get_random_action()
            self.analysis_run(self.obj, self.box_obj, region_values, force_dir_str)
            self.doc.recompute()
            self.calc_analysis()
            if (self.flag_success == 1):
                self.get_result()
                if (self.flag_save == 1):
                    self.save_result_step('./')

                self.region_values_vec.append(region_values)
                self.force_dir_str_vec.append(force_dir_str)
            else:
                self.flag_break = 1
                break

        if (self.flag_save == 1):
            pickle_dict = self.create_pickle_dict()
            self.save_result_info('./pickle_meta_data', pickle_dict)
        return self.flag_break

    def create_pickle_dict(self):
        pickle_dict = {'region_values': self.region_values_vec,
                       'force_dir_str': self.force_dir_str_vec,
                       'episode_length': self.num_actions,
                       'action_type': self.action_type,
                       'force_value': self.force_value,
                       'force_factor': self.force_factor}
        return pickle_dict

    def get_random_action(self):
        if (self.action_type == 'circle'):
            # TODO: is this optimized? maybe there is a cleaner way?
            c_circle = np.array([np.random.uniform(self.min_vals[self.column_index], self.max_vals[self.column_index], 1),
                                 np.random.uniform(self.min_vals[self.column_index + 1], self.max_vals[self.column_index + 1], 1)])
            r_circle = np.random.uniform(self.EPSILON * 10, abs(self.max_vals[self.column_index] - self.min_vals[self.column_index]), 1)
            region_values = (c_circle, r_circle)
        elif (self.action_type == 'rectangle'):
            w_rectangle = self.obj_dims[self.column_index] / self.NUM_CS
            coor_rectangle = np.random.uniform(self.min_vals[self.column_index],
                                               self.max_vals[self.column_index] - w_rectangle, 1)
            region_values = (coor_rectangle, w_rectangle)
        force_dir_str = 'top' if np.random.randint(2, size=1) else 'bottom'
        return region_values, force_dir_str

    def run_step(self, region_values=None, force_dir_str='top'):
        self.count_action += 1
        if (self.flag_first_run == 1):
            self.flag_first_run = 0
        else:
            self.reset_doc_action()
            self.initialize_obj()
            self.calc_obj_info()

        flag_valid = 0
        if (region_values == None):
            region_values, force_dir_str = self.get_random_action()
        else:
            # TODO: fix the validity check.
            # flag_valid = self.check_valid_region()
            flag_valid = 1

        if (flag_valid == 1):
            self.analysis_run(self.obj, self.box_obj, region_values, force_dir_str)
            self.doc.recompute()
            self.calc_analysis()
            if (self.flag_success == 1):
                self.get_result()

                self.region_values_vec.append(region_values)
                self.force_dir_str_vec.append(force_dir_str)
            else:
                self.flag_break = 1

        else:
            print('invalid action!')

        return self.flag_break

if __name__ == '__main__':
    # PATH = './fc1_Face109_Plus901000.ply'
    # fc_env = freecad_env(PATH)
    # fc_env.flag_save = 1
    # fc_env.run()
    # fc_env.save_result_step('./')
    PATH = ''
    fc_env = freecad_env(force_value=1e4, num_actions=5, action_type='rectangle', flag_rand_action=0)
    fc_env.flag_save = 1
    num_samples = 4 # number of episodes
    count = 0
    max_repeat_crash = 5
    save_path = './results/samples_'
    while count < num_samples:
        region_values_vec = []
        force_dir_str_vec = []
        crash_couter = 0
        while (fc_env.count_action < fc_env.num_actions):

            if (len(region_values_vec) > fc_env.count_action):
                region_values = region_values_vec[fc_env.count_action]
                force_dir_str = force_dir_str_vec[fc_env.count_action]
            else:
                region_values, force_dir_str = fc_env.get_random_action()
                region_values_vec.append(region_values)
                force_dir_str_vec.append(force_dir_str)
            flag_break = fc_env.run_step(region_values=region_values, force_dir_str=force_dir_str)

            if (flag_break == 1) and (crash_couter < max_repeat_crash):
                fc_env.clear_doc()
                fc_env.initialize_doc()
                crash_couter += 1
            elif (crash_couter >= max_repeat_crash):
                break

        if (fc_env.flag_save == 1) and (fc_env.count_action == fc_env.num_actions):
            pickle_dict = fc_env.create_pickle_dict()
            fc_env.save_result_info(save_path + 'pickle_meta_data_{:03d}'.format(count), pickle_dict)
            fc_env.save_result_all(save_path + '{:03d}'.format(count))
            count += 1
