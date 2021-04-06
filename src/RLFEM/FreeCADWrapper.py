import os
import sys

if "CONDA_PREFIX" in os.environ.keys():
    sys.path.append(os.path.join(os.environ["CONDA_PREFIX"], "lib"))

# Import FreeCAD
try:
    import FreeCAD
    import FreeCADGui
    import Part
    import ObjectsFem
    # from femguiobjects import FemGui
    # import FemGui
    # from Mod import Fem
    import Fem
    import Mesh
    import femmesh.femmesh2mesh
    from femtools import ccxtools
except:
    print("FreeCAD is not installed, skipping for now.")

import numpy as np

class FreeCADWrapper(object):
    """Class allowing to interact with the FreeCad simulation.

    TODO: more description
    """

    def __init__(self, path=''):
        """
        Args:
            path: path to the .ply file.
        """
        self.path = path
        
        # TODO: add the number of actions (sequence) and repeat until it's satisfied
        # doc = FreeCAD.getDocument("Unnamed") # CHANGE LATER!
        self.doc = App.newDocument("doc")

        if not (self.path == ''):
            self.insert_mesh()
        else:
            # TODO: correct the dimensions
            obj = self.doc.addObject('Part::Box', 'Box')
            obj.Height = 2
            obj.Width = 5
            obj.Length = 50

        self.obj = self.doc.Objects[-1]

        if (self.obj.Module == 'Mesh'):
            self.convert2solidmesh()
            self.obj = self.doc.Objects[-1]

        self.box_obj = self.doc.addObject('Part::Box', 'Box')
        # box_obj.ViewObject.Visibility = False

        self.v_CoM, self.v_NoF = self.extract_info_mesh(self.obj)

        self.min_vals = self.v_CoM.min(axis=0)
        self.max_vals = self.v_CoM.max(axis=0)
        self.obj_dims = abs(self.max_vals - self.min_vals)
        self.LENGTH = self.obj_dims[0]  # x direction
        self.WIDTH = self.obj_dims[1]  # y direction
        self.HEIGHT = self.obj_dims[2]  # z direction

        self.NUM_BIN = 1000

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
        solid_obj = self.doc.addObject("Part::Feature", shape_obj.Name + '_solid')
        solid_obj.Label = shape_obj.Name + ' (Solid)'
        solid_obj.Shape = __s__
        solid_obj.Shape = solid_obj.Shape.removeSplitter()
        del __s__
        return solid_obj


    def convert2solidmesh(self):
        shape_obj = self.doc.addObject("Part::Feature", self.obj.Name)
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
        mask_normal_1 = np.dot(self.v_NoF[self.v_CoM_sorted[mask_1, -1].astype(int), :], -normal_vector) > 0.95
        mask_normal_2 = np.dot(self.v_NoF[self.v_CoM_sorted[mask_2, -1].astype(int), :], normal_vector) > 0.95
        face_fixed_indx = np.concatenate(
            (self.v_CoM_sorted[mask_2, -1].astype(int)[mask_normal_2], self.v_CoM_sorted[mask_1, -1].astype(int)[mask_normal_1]))
        return face_fixed_indx

    def check_valid_circle(self, c_circle, r_circle):
        mask_c_circle = (abs(self.v_CoM_sorted[:, self.column_index] - c_circle[self.column_index]) < 10 * self.EPSILON)
        if not ((c_circle[self.column_index + 1] >= self.v_CoM_sorted[mask_c_circle, self.column_index + 1].min(axis=0)) & (
                c_circle[self.column_index + 1] <= self.v_CoM_sorted[mask_c_circle, self.column_index + 1].max(axis=0))):
            print('center of circle does not lie on the mesh')
            # TODO: CHECK radius as well!

    def get_faces_constraint_force(self, c_circle, r_circle, force_dir_str):

        mask_c_circle_1 = (abs(self.v_CoM_sorted[:, self.column_index] - c_circle[self.column_index]) < r_circle)
        mask_c_circle_2 = np.linalg.norm(self.v_CoM_sorted[mask_c_circle_1, self.column_index:self.column_index + 2] - c_circle,
                                         axis=1) <= r_circle
        force_direction = np.zeros(3)
        if (force_dir_str == 'top'):
            force_direction[self.column_index + 2] = 1
        elif (force_dir_str == 'bottom'):
            force_direction[self.column_index + 2] = -1
        face_indx_in_circ = self.v_CoM_sorted[mask_c_circle_1, -1].astype(int)[mask_c_circle_2]
        mask_normal = np.dot(self.v_NoF[face_indx_in_circ, :], force_direction) > 0
        return face_indx_in_circ[mask_normal]


    def add_FEM_mesh_analysis(self, obj):
        femmesh_obj = self.doc.addObject('Fem::FemMeshShapeNetgenObject', 'FEMMeshNetgen')
        femmesh_obj.Shape = obj
        femmesh_obj.MaxSize = 1
        self.doc.Analysis.addObject(femmesh_obj)


    def add_FEM_constraint_fixed(self, obj):
        self.doc.addObject("Fem::ConstraintFixed", "FemConstraintFixed")
        self.doc.FemConstraintFixed.Scale = 1
        self.doc.Analysis.addObject(self.doc.FemConstraintFixed)
        self.face_fixed_indx = self.get_faces_constraint_fixed()
        self.doc.FemConstraintFixed.References = list(
            zip([obj] * len(self.face_fixed_indx), ['Face' + str(x + 1) for x in self.face_fixed_indx]))

    def add_FEM_constraint_force(self, obj, box_obj, c_circle, r_circle, force_dir_str):
        self.doc.addObject("Fem::ConstraintForce", "FemConstraintForce")
        self.doc.Analysis.addObject(self.doc.FemConstraintForce)

        self.face_force_indx = self.get_faces_constraint_force(c_circle, r_circle, force_dir_str)

        self.doc.FemConstraintForce.Force = 1000
        self.doc.FemConstraintForce.Direction = (box_obj, ["Face6"])
        self.doc.FemConstraintForce.Reversed = True if force_dir_str == 'top' else False
        self.doc.FemConstraintForce.Scale = 100
        self.doc.FemConstraintForce.References = list(
            zip([obj] * len(self.face_force_indx), ['Face' + str(x + 1) for x in self.face_force_indx]))


    def create_FEM_solid(self):
        self.material_object = ObjectsFem.makeMaterialSolid(self.doc, "SolidMaterial")
        mat = self.material_object.Material
        mat['Name'] = "Steel-Generic"
        mat['YoungsModulus'] = "210000 MPa"
        mat['PoissonRatio'] = "0.30"
        mat['Density'] = "7900 kg/m^3"
        self.material_object.Material = mat
        # FemGui.getActiveAnalysis().addObject(material_object)
        self.doc.Analysis.addObject(self.material_object)


    def creating_analysis(self):
        self.analysis_object = ObjectsFem.makeAnalysis(FreeCAD.ActiveDocument, 'Analysis')
        self.analysis_object.addObject(ObjectsFem.makeSolverCalculixCcxTools(self.doc))

    def analysis_run(self, obj, box_obj, c_circle, r_circle, force_dir_str):
        self.creating_analysis()
        self.add_FEM_mesh_analysis(obj)
        self.add_FEM_constraint_fixed(obj)
        self.add_FEM_constraint_force(obj, box_obj, c_circle, r_circle, force_dir_str)
        self.create_FEM_solid()

    def calc_analysis(self):
        fea = ccxtools.FemToolsCcx()
        fea.update_objects()
        fea.setup_working_dir()
        fea.setup_ccx()
        message = fea.check_prerequisites()
        if not message:
            fea.purge_results()
            fea.write_inp_file()
            # on error at inp file writing, the inp file path "" was returned (even if the file was written)
            # if we would write the inp file anyway, we need to again set it manually
            # fea.inp_file_name = '/tmp/FEMWB/FEMMeshGmsh.inp'
            fea.ccx_run()
            fea.load_results()
        else:
            FreeCAD.Console.PrintError("Oh, we have a problem! {}\n".format(message))  # in report view
            print("Oh, we have a problem! {}\n".format(message))  # in python console

    def get_result(self):
        for m in self.doc.Analysis.Group:
            if m.isDerivedFrom('Fem::FemResultObject'):
                result_object = m
                break
        femmesh_obj = result_object.Mesh.FemMesh

        out_mesh = femmesh.femmesh2mesh.femmesh_2_mesh(femmesh_obj, result_object)
        self.new_mesh = Mesh.Mesh(out_mesh)

    def save_result(self, save_path):
        # TODO: add proper save_path and have num_action included in the name
        Mesh.show(self.new_mesh)
        ###5
        # export one mesh as .ply file
        __objs__ = []
        __objs__.append(self.doc.getObject("Mesh"))

        path = "./"

        filename = path + "test_01" + ".ply"
        print(__objs__)

        # Mesh.export(__objs__,u"wowply.ply")
        Mesh.export(__objs__, filename)

        del __objs__

    def run(self):
        # c_circle = np.array([15, 7])
        c_circle = np.array([40, 7])
        r_circle = 5
        force_dir_str = 'top'

        self.analysis_run(self.obj, self.box_obj, c_circle, r_circle, force_dir_str)
        self.doc.recompute()
        self.calc_analysis()
        self.get_result()

