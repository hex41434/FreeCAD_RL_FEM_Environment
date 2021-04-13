import numpy as np
import copy
import open3d

def create_mesh_open3d(vertices, faces):
    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(vertices)
    mesh.triangles = open3d.utility.Vector3iVector(faces)
    return mesh

def mesh_decimation_open3d(mesh, max_triangles=1000):
    mesh_out = copy.deepcopy(mesh)
    return mesh_out.simplify_quadric_decimation(target_number_of_triangles=max_triangles)

def get_mesh_info_open3d(mesh):
    # TODO: this is probably the worst way to create such an array. an optimized way is needed.
    faces = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    face_vertices = np.zeros((faces.shape[0], faces.shape[1] * 3))
    for i in range(faces.shape[0]):
        for j in range(faces.shape[1]):
            face_vertices[i, 3 * j: 3 * (j + 1)] = vertices[faces[i, j], :]

    return face_vertices.tolist()