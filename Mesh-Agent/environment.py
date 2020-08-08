import numpy as np
import sys
import scipy.ndimage as sn
import copy
from PIL import Image
from utils import utils
import config as p


class Environment():
    
    def __init__(self):
        
        self.vox_size=p.GT_VOX_RESO_H
        self.ref_size=p.REF_IMG_RESO
        self.action_num=p.ACTION_NUM
        self.max_step=p.MAX_STEP
        self.loop_num=p.LOOP_NUM 
        self.loop_feat_dim=p.LOOP_FEAT_DIM
        
        self.target = np.zeros((1, self.vox_size, self.vox_size, self.vox_size))
        self.ref= np.zeros((1, self.ref_size, self.ref_size))
        self.prim_v=None
        self.prim_f=None
        self.intial_prim_v=None
        
        self.target_points = np.zeros((1, 3))
        self.target_h_points = np.zeros((1, 3))

        self.name = ''
        
        self.last_IOU = 0
        self.step_count = 0
        self.step_vec=np.zeros((self.max_step), dtype=np.int)   
        
        #unit movement, relative to 1.0
        self.m_unit=0.01
        
        #action maps
        self.action_map, self.map_action = self.generate_action_map()
        
        self.transform_scale=1
        self.transform_move=0
        
    
    def reset(self, shape_infopack):
        
        #the information data of the new shape
        self.name, vox_l_fn, vox_h_fn, prim_mesh_fn, loop_info_fn, ref_type, ref_fn = shape_infopack
        
        #reset all
        self.step_count = 0
        self.step_vec=np.zeros((self.max_step), dtype=np.int)
        
        #load reference image
        img = Image.open(ref_fn)
        if ref_type == 'rgb':
            image = Image.new('RGB', size=(600, 600), color=(255,255,255))
            image.paste(img, (0, 0), mask=img)
            img=image
            
        #process and reset reference image
        img = img.convert('L')        
        img = img.resize((self.ref_size, self.ref_size), Image.ANTIALIAS)
        self.raw_img=copy.copy(img)
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        self.ref=img/255.0
        
        #load and reset primitive mesh
        self.prim_v, self.prim_f = utils.load_obj(prim_mesh_fn)
        self.init_prim_v=copy.copy(self.prim_v)

        #load and reset edgeloop info
        valid, self.ctrl_v, self.loop, self.loop_map, self.box_loop = self.load_loop(loop_info_fn)
        if valid==False or self.prim_v.shape[0]==0:
            return False, None, None, None

        #load groundtruth_data
        shape=utils.load_voxel_data(vox_h_fn).astype(np.int)
        
        #reset groundtruth
        self.target = shape
        self.target_points=np.argwhere(self.target==1)
        
        #alignment and normalization     
        c1,c2 = utils.get_corner(self.target_points)
        self.transform_scale = np.linalg.norm(c1-c2)        
        self.transform_move = c1/self.transform_scale
        self.ctrl_v=self.ctrl_v/self.transform_scale-self.transform_move
        self.prim_v=self.prim_v/self.transform_scale-self.transform_move
        
        #reset initial IOU
        self.last_IOU = self.compute_IOU(self.prim_v)
        
        ctrl_info=np.zeros((self.loop_num, 2, p.LOOP_FEAT_DIM))
        ctrl_info[:,:,0:3] = self.ctrl_v
        ctrl_info[:,0,3] = self.loop_map[:,0]
        ctrl_info[:,1,3] = self.loop_map[:,0]
        
        return valid, self.ref, ctrl_info, self.step_vec
    
    def generate_action_map(self):
        #map between action id and detailed operation parameters
        action_id=0
        
        #map action id to loop_id (10), control_point (2), direction (3), range[-3,3] (6)
        action_map=np.zeros((10*2*3*6,4), dtype=int)
        map_action=np.zeros((10,2,3,6), dtype=int)        
        for i in range(10):
            for j in range(2):
                for k in range(3):
                    for l in range(6):                        
                        map_action[i][j][k][l]=action_id
                        
                        l_ = l + 1
                        if l>=3:
                            l_ = l-6
                            
                        para=np.array([i,j,k,l_])
                        action_map[action_id]=para
                        
                        action_id+=1
                        
        return action_map, map_action
    
    
    def get_valid_action_mask(self):
        
        valid_mask=np.ones((self.action_num),dtype=np.int)
        for a in range(self.action_num):
            
            i, j, k, l = self.action_map[a]
            loop_id=self.step_count%self.loop_num
            
            if i!=loop_id:
                valid_mask[a]=-10000.0
            
        return valid_mask
    
    
    def load_loop(self, path):
        
        fopen = open(path, 'r', encoding='utf-8')
        lines = fopen.readlines()
        linecount = 0
        
        loops=[]
        maps = np.zeros((self.loop_num,4), np.int)
        ctrl_v = np.zeros((self.loop_num,2,3), np.float)
        loops = np.zeros((self.loop_num,4), np.int)
        boxloop=[]
        
        for line in lines:
            
            word = line.split()
            
            if word[0]=='#':
                continue
            
            if linecount == 0:
                l_num = int(word[0])
                box_num = int(word[1])
            
            if l_num>self.loop_num:
                return False, None, None, None, None
            
            if linecount>=1 and linecount< 1+self.loop_num:
                id=linecount-1
                ctrl_v[id][0]=np.array(word[0:3])
                ctrl_v[id][1]=np.array(word[3:6])
                
            if linecount >= 1+self.loop_num and linecount< 1+self.loop_num*2:
                id=linecount-(1+self.loop_num)
                loops[id]=np.array(word[0:4])

            if linecount >= 1+self.loop_num*2 and linecount< 1+self.loop_num*3:
                id=linecount-(1+self.loop_num*2)
                maps[id]=np.array(word[0:4])
            
            if linecount >=1+self.loop_num*3:
                id=linecount-(1+self.loop_num*3)
                boxloopid=np.array(word).tolist()
                boxloop.append(boxloopid)
            
            linecount = linecount + 1

        fopen.close()
        
        return True, ctrl_v, loops, maps, boxloop
    
    
    def compute_IOU(self, prim_v):
        
        prim_v=((prim_v+self.transform_move)*self.transform_scale).astype(np.int)
        prim_v=np.clip(prim_v,0,self.vox_size-1)
        canvas=np.zeros_like(self.target, dtype=np.int)
        
        #for each box
        for i in range(len(self.box_loop)):
            
            #for each loop of this box
            for j in range(len(self.box_loop[i])-1):
                
                l_id1=int(self.box_loop[i][j])
                l_id2=int(self.box_loop[i][j+1])

                c1 = prim_v[self.loop_map[l_id1][2]]
                c1_= prim_v[self.loop_map[l_id1][3]]+1
                
                c2 = prim_v[self.loop_map[l_id2][2]]
                c2_= prim_v[self.loop_map[l_id2][3]]+1
                
                #loop orthogonal to k-axis
                k=self.loop_map[l_id1][0]
                #number of intervals
                d=int(c2[k]-c1[k]) 
                #k axis intervals index
                t=np.linspace(c1[k], c2[k], d).astype(np.int)
                
                if k==0:
                    p=np.linspace(c1[1], c2[1], d).astype(np.int)
                    p_=np.linspace(c1_[1],c2_[1], d).astype(np.int)
                    q=np.linspace(c1[2], c2[2], d).astype(np.int)
                    q_=np.linspace(c1_[2],c2_[2], d).astype(np.int)
                    for n in range(d):
                        canvas[t[n], p[n]:p_[n], q[n]:q_[n]]=1
                        
                if k==1:
                    p=np.linspace(c1[0], c2[0], d).astype(np.int)
                    p_=np.linspace(c1_[0],c2_[0], d).astype(np.int)
                    q=np.linspace(c1[2], c2[2], d).astype(np.int)
                    q_=np.linspace(c1_[2], c2_[2], d).astype(np.int)
                    for n in range(d):
                        canvas[p[n]:p_[n], t[n], q[n]:q_[n]]=1
                    
                if k==2:
                    p=np.linspace(c1[0], c2[0], d).astype(np.int)
                    p_=np.linspace(c1_[0],c2_[0], d).astype(np.int)
                    q=np.linspace(c1[1], c2[1], d).astype(np.int)
                    q_=np.linspace(c1_[1],c2_[1], d).astype(np.int)
                    for n in range(d):
                        canvas[p[n]:p_[n], q[n]:q_[n],t[n]]=1
        
        intersect=canvas & self.target
        i_count=np.sum(intersect == 1)
        
        union=canvas | self.target
        u_count=np.sum(union == 1)
        
        iou=float(i_count)/float(u_count)
        
        return iou
    

    def edit_loop(self, action):
    
        i,j,k,l=self.action_map[action]
        
        ctrl_v=copy.copy(self.ctrl_v)
        prim_v=copy.copy(self.prim_v)
        
        loop_axis, loop_id, vid1, vid2=self.loop_map[i]
        
        if loop_axis==k:
            return prim_v, ctrl_v
        
        old_keyv=copy.copy(ctrl_v[i])
        new_keyv=copy.copy(ctrl_v[i])
        
        #edit the control point
        new_keyv[j][k]+=l*self.m_unit
        ctrl_v[i][j][k]+=l*self.m_unit
        
        if new_keyv[j][k]<0:
            new_keyv[j][k]=0
            ctrl_v[i][j][k]=0
        elif new_keyv[j][k]>1:
            new_keyv[j][k]=1
            ctrl_v[i][j][k]=1
        
        #avoid penetration by a safe distance
        if new_keyv[0][k]+2.0*self.m_unit > new_keyv[1][k]:
            if j==0:
                new_keyv[j][k]=new_keyv[j+1][k]-2.0*self.m_unit
            elif j==1:
                new_keyv[j][k]=new_keyv[j-1][k]+2.0*self.m_unit
            ctrl_v[i][j][k]=new_keyv[j][k]
        
        #edit the whole loop by scaling
        loop_indices=self.loop[loop_id]        
        old_selected_loop_v = prim_v[loop_indices]
        new_selected_loop_v=utils.scale_vertex_by_bbox(old_selected_loop_v, old_keyv[0], old_keyv[1], new_keyv[0], new_keyv[1])
        prim_v[loop_indices,:]=new_selected_loop_v
        
        return prim_v, ctrl_v
    
    
    def next(self, action):
        # print(self.action_map[action])
        self.prim_v, self.ctrl_v = self.edit_loop(action)
        IOU=self.compute_IOU(self.prim_v)
        reward=IOU-self.last_IOU
        self.last_IOU=IOU
        
        self.step_vec=np.zeros((self.max_step), dtype=np.int)
        
        if self.step_count == self.max_step:
            done=True
        else:
            done=False
            self.step_vec[self.step_count]=1
        
        self.step_count+=1
        
        
        ctrl_info=np.zeros((self.loop_num,2, self.loop_feat_dim))
        ctrl_info[:,:,0:3] = self.ctrl_v
        ctrl_info[:,0,3] = self.loop_map[:,0]
        ctrl_info[:,1,3] = self.loop_map[:,0]
        
        return self.ref, ctrl_info, self.step_vec, reward, done
    
    
    def next_no_update(self, action):

        try_prim_v, try_ctrl_v = self.edit_loop(action)
        IOU=self.compute_IOU(try_prim_v)
        reward=IOU-self.last_IOU
        step_vec=np.zeros((self.max_step), dtype=np.int)

        if self.step_count+1 >= self.max_step:
            done=True
        else:
            done=False
            step_vec[self.step_count+1]=1
        
        ctrl_info=np.zeros((self.loop_num, 2, self.loop_feat_dim))
        ctrl_info[:,:,0:3] = try_ctrl_v
        ctrl_info[:,0,3] = self.loop_map[:,0]
        ctrl_info[:,1,3] = self.loop_map[:,0]
        
        return self.ref, ctrl_info, step_vec, reward, done
    
    
    def get_virtual_expert_action(self, valid_mask, random=False):
        
        if not random:
            loop_id=self.step_count%self.loop_num
        else:
            loop_id=np.random.randint(0,self.loop_num)
        
        max_action=-1
        max_reward=-1000
        
        for j in range(2):
            for k in range(3):
                for l in range(6):
                    
                    action=self.map_action[loop_id, j, k, l]
                    
                    if valid_mask[action]==0:
                        continue
                    
                    s_, loop_, step, reward, done = self.next_no_update(action)
                    
                    if reward>max_reward:
                        max_reward=reward
                        max_action=action
        
        return max_action
    
                            
    def output_result(self, log_info, save_path):
        
        #save input shape reference
        self.raw_img.save(save_path + str(self.name) + '.png')
        
        #output file name
        output_name= save_path + log_info + ".obj" 
        
        #output result
        output_prim_v=copy.copy(self.prim_v)
        output_ctrl_v=copy.copy(self.ctrl_v)
        output_v=(output_prim_v+self.transform_move)*self.transform_scale
        output_f=self.prim_f
        utils.save_mesh_obj(output_v, output_f, output_name)
        
        return output_v, output_f
        
    
    def eval_CD_error(self, output_v, output_f, GT_filename):
    
        output_f=utils.triangulation_quad_mesh(output_f)
        initial_v=utils.rand_sample_points_on_tri_mesh(self.init_prim_v, output_f, 4000)
        initial_v=utils.normalize_points(initial_v)
        
        sample_v=utils.rand_sample_points_on_tri_mesh(output_v, output_f, 4000)
        sample_v=utils.normalize_points(sample_v)

        gt_v=utils.load_ply(GT_filename)
        gt_v=utils.normalize_points(gt_v)
        
        cd1_prim=utils.compute_chamfer_distance(initial_v, gt_v)
        cd2_prim=utils.compute_chamfer_distance(gt_v, initial_v)
        
        cd1_mesh=utils.compute_chamfer_distance(sample_v, gt_v)
        cd2_mesh=utils.compute_chamfer_distance(gt_v, sample_v)
        
        return cd1_prim + cd2_prim, cd1_mesh + cd2_mesh
        
        
    
        
    
        
        


