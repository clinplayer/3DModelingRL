import numpy as np
import torch
import math
import copy
import random
import os
from queue import Queue
from utils import binvox_rw

def check_dirs(dir_list):
    for dir in dir_list:
        if not os.path.exists(dir):
            os.makedirs(dir)
            print(dir+' does not exist. Created.')
            
def load_filelist(path):
    fopen = open(path, 'r', encoding='utf-8')
    lines = fopen.readlines()
    
    data_num=len(lines)
    filelist=[]
    
    linecount=0
    for line in lines:
        filelist.append(line.strip('\n'))
        linecount = linecount + 1
        
    fopen.close()
    
    return filelist

def load_off(path):
    fopen = open(path, 'r', encoding='utf-8')
    lines = fopen.readlines()
    linecount = 0
    pts = np.zeros((1,3), np.float64)
    faces = np.zeros((1,4), np.int)
    p_num=0
    f_num=0

    for line in lines:
        linecount = linecount + 1
        word = line.split()

        if linecount == 1:
            continue

        if linecount == 2:
            p_num = int(word[0])
            f_num = int(word[1])
            pts = np.zeros((p_num,3), np.float)
            faces = np.zeros((f_num, 4), np.int)

        if linecount >= 3 and linecount< 3+p_num:
            pts[linecount-3, :] = np.float64(word[0:3])
        if linecount >=3+p_num:
            faces[linecount-3-p_num] = np.int32(word[1:5])

    fopen.close()
    return pts, faces

def load_obj(path):
    fopen = open(path, 'r', encoding='utf-8')
    lines = fopen.readlines()
    pts = []
    faces = []

    for line in lines:
        word = line.split()
        if word[0]=='v':
            pts.append(np.float64(word[1:4]))
        
        if word[0]=='f':
            faces.append(np.int32(word[1:5]))
            
    fopen.close()
    
    pts=np.array(pts)
    faces=np.array(faces)
    
    return pts, faces

def load_ply(pc_filepath):
    
    fopen = open(pc_filepath, 'r', encoding='utf-8')
    lines = fopen.readlines()
    linecount=0
    
    pts=np.zeros((1,3),np.float64)

    total_point=0
    sample_interval=0
    feed_point_count=0

    for line in lines:
        linecount=linecount+1
        word=line.split()

        if linecount==4:
            total_point=int(word[2])
            pts=np.zeros((total_point,3), np.float64) 
            continue

        if linecount>13:
            pts[feed_point_count, :] = np.float64(word[0:3])
            feed_point_count+=1

    fopen.close()
    return pts
    
def load_voxel_data(path):
    with open(path, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
        return model.data

def save_list(list_data, path):
    with open(path, "w") as file:
        for data in list_data:
            file.write(str(data)+"\n")
        
def save_point_off(points, colors, path, use_color=True):
    with open(path, "w") as file:
        file.write("COFF\n")
        file.write(str(int(points.shape[0])) + " 0" + " 0\n")
        for i in range(points.shape[0]):
            file.write(str(float(points[i][0])) + " " + str(float(points[i][1])) + " " + str(float(points[i][2])) + " ")
            if use_color == False:
                file.write("150 0 0\n")
            else:
                file.write(str(colors[0]) + " " + str(colors[1]) + " " + str(colors[2]))

def save_mesh_obj(vertexlist, facelist, path):
    with open(path, "w") as file:

        for i in range(vertexlist.shape[0]):
            file.write("v "+str(vertexlist[i][0]) + " " + str(vertexlist[i][1]) + " " + str(vertexlist[i][2]) + "\n")
            
        for i in range(facelist.shape[0]):
            file.write("f " + str(int(facelist[i][0])) + " " + str(int(facelist[i][1])) + " " + \
            str(int(facelist[i][2]))+ " "+ str(int(facelist[i][3])) + " \n")
                
def get_corner(v):
    minx,miny,minz=v.min(axis=0)
    maxx,maxy,maxz=v.max(axis=0)

    corner1=np.array([minx,miny,minz])
    corner2=np.array([maxx,maxy,maxz])
    return corner1, corner2

    
def clean_boxlist(boxlist):
    newboxlist=np.zeros((boxlist.shape[0],6))
    validcount=0
    for i in range(boxlist.shape[0]):
        if boxlist[i,3]-boxlist[i,0]<=0 or boxlist[i,4]-boxlist[i,1]<=0 or boxlist[i,5]-boxlist[i,2]<=0:
            continue
        dia_len=np.linalg.norm(boxlist[i,3:6]-boxlist[i,0:3])
        if dia_len<=4:
            continue
        newboxlist[validcount]=boxlist[i,:]
        validcount+=1

    boxnum=validcount
    boxlist=newboxlist[0:boxnum,:]
    return boxlist
    
    
def clean_boxlist_notdel(boxlist):
    newboxlist=np.zeros((boxlist.shape[0],6))
    validcount=0
    for i in range(boxlist.shape[0]):
        if boxlist[i,3]-boxlist[i,0]<=0 or boxlist[i,4]-boxlist[i,1]<=0 or boxlist[i,5]-boxlist[i,2]<=0:
            boxlist[i,0:3]=boxlist[i,3:6]
            # boxlist[i,0:6]=0
        dia_len=np.linalg.norm(boxlist[i,3:6]-boxlist[i,0:3])
        if dia_len<=4:
            boxlist[i,0:3]=boxlist[i,3:6]
            # boxlist[i,0:6]=0
        newboxlist[validcount]=boxlist[i,:]
        validcount+=1

    boxnum=validcount
    boxlist=newboxlist[0:boxnum,:]
    return boxlist
    

def merge_boxlist(boxlist, canvas_size, max_box_num, merge_t, keep_color=False):
    
    #build the merging graph
    merged_box=[]
    merged_color=[]
    graph=np.zeros((boxlist.shape[0],boxlist.shape[0]), dtype=np.int)
    for i in range(boxlist.shape[0]):
        box_i=np.zeros((canvas_size,canvas_size,canvas_size),dtype=np.int)
        ci=boxlist[i].astype(np.int)
        box_i[ci[0]:ci[3],ci[1]:ci[4],ci[2]:ci[5]]=1
        
        for j in range(i+1,boxlist.shape[0]):
            box_j=np.zeros((canvas_size,canvas_size,canvas_size),dtype=np.int)
            cj=boxlist[j].astype(np.int)
            
            if (ci[0:3]==ci[3:6]).any() or (cj[0:3]==cj[3:6]).any():
                continue
            
            box_j[cj[0]:cj[3],cj[1]:cj[4],cj[2]:cj[5]]=1
            
            box_union = box_i | box_j
            
            box_merge=np.zeros((canvas_size,canvas_size,canvas_size),dtype=np.int)
            bps=np.zeros((4,3))
            bps[0],bps[1],bps[2],bps[3]=ci[0:3],ci[3:6],cj[0:3],cj[3:6]            
            nc1,nc2=get_corner(bps)
            nc1=nc1.astype(np.int)
            nc2=nc2.astype(np.int)
            
            box_merge[nc1[0]:nc2[0], nc1[1]:nc2[1], nc1[2]:nc2[2]]=1
            
            
            union_count=np.sum(box_union==1)
            merge_count=np.sum(box_merge==1)
            
            ratio=float(union_count)/float(merge_count+1)
            
            if ratio>merge_t:
                graph[i][j]=1
                graph[j][i]=1
            
    #components
    
    visit=np.zeros((boxlist.shape[0]),dtype=np.int)

    for i in range(boxlist.shape[0]):

        q = Queue()
        group=[]
        
        max_index=i #to record the color
        max_vol=0
        
        if visit[i]==1:
            continue
        
        visit[i]=1
        q.put(i)
        group.append(i)
        
        while not q.empty():
            now=q.get()
            
            sides=boxlist[now,3:6]-boxlist[now,0:3]
            vol=sides[0]*sides[0] + sides[1]*sides[1] + sides[2]*sides[2]
            if vol>=max_vol:
                max_vol=vol
                max_index=now
                
            for j in range(boxlist.shape[0]):
                if graph[now,j]==1 and visit[j]==0:
                    q.put(j)
                    visit[j]=1
                    group.append(j)
        
        groupsize=len(group)
        points=np.zeros((groupsize*2,3))
        count=0
        for i in group:
            points[count,:]=boxlist[i,0:3]
            points[count+1,:]=boxlist[i,3:6]
            count+=2
            
        c1,c2=get_corner(points)
        box=np.array([c1[0],c1[1],c1[2],c2[0],c2[1],c2[2]])
        merged_box.append(box)
        merged_color.append(max_index)
    
    
    merged_box=np.array(merged_box)
    final_box=copy.copy(merged_box)
    if keep_color:
        final_box=np.zeros((max_box_num,6),np.float)
        for i in range(len(merged_color)):
            box_id=merged_color[i]
            final_box[box_id]=merged_box[i]
    
    return final_box
       
       
def sort_boxlist(boxlist):
    box_num=boxlist.shape[0]
    edge_len=boxlist[:,3:6]-boxlist[:,0:3]
    dis_to_o= np.linalg.norm((boxlist[:,3:6]+boxlist[:,0:3])*0.5, axis=1)
    max_len=edge_len.max(axis=1)
    max_axis=edge_len.argmax(axis=1)
    xbox=[]
    ybox=[]
    zbox=[]

    for i in range(box_num):
    
        if max_axis[i]==0:
            box=np.zeros((7))
            box[0:6]=boxlist[i]
            box[6]=dis_to_o[i]
            xbox.append(box)
            
        elif max_axis[i]==1:
            box=np.zeros((7))
            box[0:6]=boxlist[i]
            box[6]=dis_to_o[i]
            ybox.append(box)
        
        elif max_axis[i]==2:
            box=np.zeros((7))
            box[0:6]=boxlist[i]
            box[6]=dis_to_o[i]
            zbox.append(box)
    
    xbox=np.array(xbox)
    ybox=np.array(ybox)
    zbox=np.array(zbox)
    
    boxes=[]
    if len(xbox)!=0:
        xbox = xbox[xbox[:, 6].argsort()]
        for i in range(len(xbox)):
            boxes.append(xbox[i,0:6])
    
    if len(ybox)!=0:
        ybox = ybox[ybox[:, 6].argsort()]
        for i in range(len(ybox)):
            boxes.append(ybox[i,0:6])
    
    if len(zbox)!=0:
        zbox = zbox[zbox[:, 6].argsort()] 
        for i in range(len(zbox)):
            boxes.append(zbox[i,0:6])
    
    boxlist=np.array(boxes)
        
    return boxlist
       
       
def save_boxlist(boxlist, path, colors):

    boxnum=boxlist.shape[0]
    facelist = np.zeros((12 * boxnum, 3))
    
    with open(path, "w") as file:
        file.write("OFF\n")
        file.write(str(8 * boxnum) + " " + str(12 * boxnum) + " " + str(0) + "\n")

        for i in range(boxnum):
            x, y, z, x_, y_, z_ = boxlist[i][0], boxlist[i][1], boxlist[i][2], boxlist[i][3], boxlist[i][4], boxlist[i][5]
            v = np.zeros((8, 3), np.float16)
            v[0] = [x, y, z]
            v[1] = [x_, y, z]
            v[2] = [x_, y, z_]
            v[3] = [x, y, z_]
            v[4] = [x, y_, z_]
            v[5] = [x, y_, z]
            v[6] = [x_, y_, z]
            v[7] = [x_, y_, z_]
            for j in range(8):
                file.write(str(v[j][0]) + " " + str(v[j][1]) + " " + str(v[j][2]) + "\n")

            f = np.zeros((12, 3))
            f[0] = [0, 1, 2]
            f[1] = [0, 2, 3]
            f[2] = [0, 3, 4]
            f[3] = [0, 4, 5]
            f[4] = [0, 5, 6]
            f[5] = [0, 6, 1]
            f[6] = [7, 5, 4]
            f[7] = [7, 6, 5]
            f[8] = [7, 1, 6]
            f[9] = [7, 2, 1]
            f[10] = [7, 3, 2]
            f[11] = [7, 4, 3]
            for j in range(12):
                facelist[i * 12 + j, :] = f[j] + i * 8

        for i in range(boxnum):
            for j in range(12):
                file.write(
                    "3 " + str(int(facelist[i * 12 + j][0])) + " " + str(int(facelist[i * 12 + j][1])) + " " + str(
                        int(facelist[i * 12 + j][2])) + " ")
                file.write(str(int(colors[i][0])) + " " + str(int(colors[i][1])) + " " + str(int(colors[i][2])) + "\n")

                    
def scale_vertex_by_bbox(v, source1, source2, target1, target2):
    
    s, s_ = source1, source2
    t, t_ = target1, target2
    cs=0.5*(source1+source2)
    ct=0.5*(target1+target2)
    move=t-s
    
    dt=t-t_
    ds=s-s_
    for i in range(3):
        if ds[i]==0:
            ds[i]=1
    
    scale=np.true_divide(dt ,ds)
    cs=cs*scale
    move=ct-cs
    
    v=v*scale+move
    
    return v


def get_scaled_prim_point(boxlist, source_v, target_v):
    
    c_source_1, c_source_2 = get_corner(source_v)
    c_target_1, c_target_2 = get_corner(target_v)
    
    box_v1=boxlist[:,0:3]
    box_v2=boxlist[:,3:6]
    
    box_v1=scale_vertex_by_bbox(box_v1, c_source_1, c_source_2, c_target_1, c_target_2)
    box_v2=scale_vertex_by_bbox(box_v2, c_source_1, c_source_2, c_target_1, c_target_2)
    
    box=np.concatenate((box_v1,box_v2), axis=1)
    
    return box

def compute_dis_point_to_box(v, box):
    
    #on the boundary or inside
    if v[0]>=box[0] and v[0]<=box[3] and v[1]>=box[1] and v[1]<=box[4] and v[2]>=box[2] and v[2]<=box[5]:
        dis=v[0]-box[0]
        dis=min([dis, box[3]-v[0]])
        dis=min([dis, v[1]-box[1]])
        dis=min([dis, box[4]-v[1]])
        dis=min([dis, v[2]-box[2]])
        dis=min([dis, box[5]-v[2]])
        return -dis
    
    cp=np.zeros((3),np.float)
    for i in range(3):
        if v[i]<box[i]:
            cp[i]=box[i]
            
        elif v[i]>box[i+3]:
            cp[i]=box[i+3]
            
        else:
            cp[i]=v[i]
    
    lv=v-cp
    dis=np.linalg.norm(lv)
    
    return dis

    
def compute_face_areas(vertices, faces):

    v0 = vertices[faces[:, 0], :]
    v1 = vertices[faces[:, 1], :]
    v2 = vertices[faces[:, 2], :]
    tmp_cross = np.cross(v0 - v2, v1 - v2)

    areas = 0.5 * np.sqrt(np.sum(tmp_cross * tmp_cross, axis=1))
    return areas


def rand_sample_points_on_tri_mesh(vertices, faces, num_sample):
    areas = compute_face_areas(vertices, faces)
    probabilities = areas / areas.sum()
    weighted_random_indices = np.random.choice(range(areas.shape[0]), size=num_sample, p=probabilities)

    u = np.random.rand(num_sample, 1)
    v = np.random.rand(num_sample, 1)
    w = np.random.rand(num_sample, 1)

    sum_uvw = u + v + w
    u = u / sum_uvw
    v = v / sum_uvw
    w = w / sum_uvw

    v0 = vertices[faces[:, 0], :]
    v1 = vertices[faces[:, 1], :]
    v2 = vertices[faces[:, 2], :]
    v0 = v0[weighted_random_indices]
    v1 = v1[weighted_random_indices]
    v2 = v2[weighted_random_indices]

    sampled_v = (v0 * u) + (v1 * v) + (v2 * w)
    sampled_v = sampled_v.astype(np.float32)
    
    return sampled_v


def normalize_points(vertices, max_size=1):
    points_max = np.max(vertices, axis=0)
    points_min = np.min(vertices, axis=0)
    vertices_center = (points_max + points_min) / 2
    points = vertices - vertices_center[None, :]
    max_radius = np.max(np.sqrt(np.sum(points * points, axis=1)))
    vertices = points / max_radius * max_size / 2.0
    return vertices


def triangulation_quad_mesh(face):
    tri_face=np.zeros((2*face.shape[0],3), np.int)
    for i in range(face.shape[0]):
        tri_face[i*2]=np.array([face[i][3], face[i][2], face[i][1]], np.int)-1
        tri_face[i*2+1]=np.array([face[i][1], face[i][0], face[i][3]], np.int)-1
    return tri_face

def compute_chamfer_distance(p1, p2):
    
    p1=torch.from_numpy(p1[None,:,:]).double()
    p2=torch.from_numpy(p2[None,:,:]).double()
    
    assert p1.size(0) == 1 and p2.size(0) == 1
    assert p1.size(2) == p2.size(2)

    p1 = p1.repeat(p2.size(1), 1, 1)
    p1 = p1.transpose(0, 1)
    p2 = p2.repeat(p1.size(0), 1, 1)
    dist = torch.add(p1, torch.neg(p2))
    dist = torch.norm(dist, 2, dim=2)
    dist = torch.min(dist, dim=1)[0]

    dist = torch.mean(dist)

    return dist

