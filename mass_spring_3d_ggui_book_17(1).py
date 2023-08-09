import taichi as ti
import numpy as np
from plyfile import PlyData, PlyElement
import trimesh
from scipy.spatial import Delaunay
    
import psutil
import os
import open3d as o3d
import time
import copy

# arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
# ti.init(arch)
ti.init(arch=ti.cpu)

def load_mesh(path):
    thresh = 3.0
    
    global substeps, n_vertexs, vertexs, faces, max_cols
    global xin, x, v, xfin, xp, vp
    global colors
    
    ########
    plydata = PlyData.read(path)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    mesh = trimesh.load(path)
    colors = mesh.visual.vertex_colors
    x = np.reshape(x, (len(x),1))
    y = np.reshape(y, (len(y),1))
    z = np.reshape(z, (len(z),1))

    vertexs = np.concatenate((x, y, z), axis = 1)
    vertexs = np.reshape(vertexs, (len(plydata['vertex']['x']),3))

    u = (vertexs[:,0] - min(vertexs[:,0]))
    v = (vertexs[:,1] - min(vertexs[:,1]))
    z = (vertexs[:,2] - min(vertexs[:,2]))

    x = u
    y = v
    tri = Delaunay(np.array([u,v]).T)
    
    
    
    global neighbor, neighbor_arr
    global top, floor
    
    
    n_vertexs = len(vertexs)
    floor = min(vertexs[:,2])*0.001-1e-7
    top = max(vertexs[:,2])*0.001+1e-7
    
    # return n_vertexs, floor, top
    
    
    
# def create_neighbors():
    faces = tri.simplices

    mesh = trimesh.Trimesh(vertices=vertexs, faces=faces)

    edges = mesh.edges_unique
    lengths = mesh.edges_unique_length

    mask = edges[np.where(lengths <= thresh)[0]]

    neighbors = {p: [] for p in range(len(vertexs))}
    # neighbors = np.array([])
    for edge in mask:
        p1, p2 = edge
        neighbors[p1].append(p2)
        neighbors[p2].append(p1)

    max_cols = max(len(row) for row in neighbors.values())

    # 创建空数组
    neighbor_arr = np.zeros((n_vertexs, max_cols), dtype=np.int32)

# 填充数组
    for i, row in enumerate(neighbors.values()):
        for j, value in enumerate(row):
            neighbor_arr[i, j] = value
            if len(row) < max_cols:
                neighbor_arr[i, len(row):] = -1
                
# def initialize_var(): 
    global gravity, spring_Y, dashpot_damping, drag_damping, dt
                 
    dt = 0.00002
    substeps = 500
    
    
    
    gravity = ti.Vector([0, 0, 0])
    spring_Y = 1000
    dashpot_damping = 3e5
    drag_damping = 0.01

    xin = ti.Vector.field(3, dtype=float, shape=(n_vertexs))
    xfin = ti.Vector.field(3, dtype=float, shape=(n_vertexs))

    x = ti.Vector.field(3, dtype=float, shape=(n_vertexs))
    v = ti.Vector.field(3, dtype=float, shape=(n_vertexs))

    xp = ti.Vector.field(3, dtype=float, shape=(1, ))
    vp = ti.Vector.field(3, dtype=float, shape=(1, ))

    neighbor = ti.Vector.field(int(max_cols), dtype=int, shape=(n_vertexs))

def initialize_mass_points():
    xin.from_numpy(vertexs*0.001)
    x.from_numpy(vertexs*0.001)
    neighbor.from_numpy(neighbor_arr)
    xp[0][2] = top
    

@ti.kernel
def substep():   
    for i in ti.ndrange(n_vertexs):
        force = ti.Vector([0.0, 0.0, 0.0]) 
        
        if x[i][2] < floor:  # Bottom and left
            x[i][2] = floor  # move particle inside
            v[i][2] = 0  # stop it from moving further
            
        if x[i][2] > xp[0][2]:  # Bottom and left
            x[i][2] = xp[0][2]  # move particle inside
            v[i][2] = 0  # stop it from moving further
               
        for k in ti.static(range(max_cols)):
            
            j = neighbor[i][k]
            if j != -1:
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                ori_ij = xin[i] - xin[j]
                d = x_ij.normalized()
                current_dist = x_ij.norm()
                original_dist = ori_ij.norm()

                # Spring force
                force += -spring_Y * d * (current_dist / original_dist - 1) * current_dist * 10
                # # # Dashpot damping
                force += -v_ij.dot(d) * d * dashpot_damping * drag_damping

        v[i] += force * dt
        x[i] += dt * v[i]

                  
    f = ti.Vector([0, 0, -10000.0])
    vp[0] += dt * f
    xp[0] += vp[0] * dt
    if xp[0][2] < floor:  # Bottom and left
        xp[0][2] = floor # move particle inside
        vp[0][2] = 0  # stop it from moving further
        
          
@ti.kernel
def update_vertices():
    for i in ti.ndrange(n_vertexs):
        xfin[i] = x[i]


def main():
    path = 'E:\\taichi\\taichiCourse01_final_hw-main\\taichiCourse01_final_hw-main\\data\\models\\paper_plan\\case\\case\\temp\\interpcd_nomesh.ply'    

    
    load_mesh(path)
        
    # create_neighbors()
    
    # initialize_var()
    
    current_t = 0.0
    initialize_mass_points()


    for i in range(substeps):#########fps
        substep()
        current_t += dt
    update_vertices()

if __name__ == '__main__':    
    #######

    time0 = time.time()

    main()
    
    print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
    xini = xin.to_numpy()*1000
    xfina = xfin.to_numpy()*1000
    # ply_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    # data = np.empty(len(xfina), dtype=ply_dtype)
    # data['x'], data['y'], data['z'] = xfina[:, 0], xfina[:, 1], xfina[:, 2]
    # data['red'], data['green'], data['blue'] = colors[:, 0], colors[:, 1], colors[:, 2]
    # vertex_element = PlyElement.describe(data, 'vertex')
    # meshio.write("mesh.ply", meshio.Mesh(points=vertexs, cells={"triangle": faces}), binary=False)
    # ply_data = PlyData([vertex_element],text=True)
    # with open('output_0.ply', 'wb') as f:
    #     ply_data.write(f)

    current_path = os.path.abspath(os.path.dirname(__file__)+os.path.sep+'..')
    f=open(current_path + '/config.txt','r')
    lines=f.readlines()
    f.close()
    conf=[]
    for l in lines:
        r=l.split(',')[-1]
        conf.append(r)

    filename=conf[1][:-1]
    filename=filename.replace('\\','/')
    path=lines[2].split(',')[1][:-1]
    path=path.replace('\\','/')
    path=path.strip('\r')
    vin=xini.reshape(-1,3)
    xyzi=copy.deepcopy(vin)
    vfin=xfina.reshape(-1,3)
    xyzf=copy.deepcopy(vfin)

    pcdi=o3d.geometry.PointCloud()
    pcdi.points=o3d.utility.Vector3dVector(xyzi)
        
    pcdfi=o3d.geometry.PointCloud()
    pcdfi.points=o3d.utility.Vector3dVector(xyzf)

    o3d.io.write_point_cloud(path+'/interpcd_i.ply',pcdi,write_ascii= True)
    o3d.io.write_point_cloud(path+'/interpcd_f.ply',pcdfi,write_ascii= True)

    print(time.time()-time0)