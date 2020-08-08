import numpy as np
import torch
import scipy.ndimage as sn
import copy
from PIL import Image
from utils import utils
import config as p


class Environment():

    def __init__(self):
    
        super(Environment, self).__init__()
        
        self.vox_size_l=p.GT_VOX_RESO_L
        self.vox_size_h=p.GT_VOX_RESO_H
        self.ref_size=p.REF_IMG_RESO
        self.box_num=p.BOX_NUM
        self.action_num=p.ACTION_NUM
        self.max_step=p.MAX_STEP
        self.loop_num=p.LOOP_NUM
        
        self.ref= np.zeros((1, self.ref_size, self.ref_size))
        self.target = np.zeros((1, self.vox_size_l, self.vox_size_l, self.vox_size_l))
        self.target_h = np.zeros((1, self.vox_size_l, self.vox_size_l, self.vox_size_l))        
        self.target_points = np.zeros((1, 3))
        self.target_h_points = np.zeros((1, 3))

        self.init_boxes = self.initialize_box()
        self.all_boxes = copy.copy(self.init_boxes)
        
        self.name = ''
        
        self.last_IOU = 0
        self.last_delete = 0
        self.last_local_IOU = 0
        self.step_count = 0
        self.step_vec=np.zeros((self.max_step), dtype=np.int)  
        
        #unit movement, relative to vox_size_l
        self.m_unit=1.0
        
        self.action_map, self.map_action = self.generate_action_map()
       
        self.colors = np.array(p.COLORS)

      
      
    def reset(self, shape_infopack):
        
        #the information data of the new shape
        self.name, vox_l_fn, vox_h_fn, ref_type, ref_fn = shape_infopack
        
        #reset all
        self.all_boxes = copy.copy(self.init_boxes)
        self.last_IOU = 0
        self.last_local_IOU = 0
        self.last_delete = 0
        self.step_count = 0
        self.step_vec=np.zeros((self.max_step), dtype=np.int)
        
        #load reference image
        img = Image.open(ref_fn)
        if ref_type=='rgb':
            image = Image.new('RGB', size=(600, 600), color=(255,255,255))
            image.paste(img, (0, 0), mask=img)
            img = image
        
        #process and reset reference image
        img = img.convert('L')        
        img = img.resize((self.ref_size, self.ref_size), Image.ANTIALIAS)
        self.raw_img=copy.copy(img)
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        self.ref=img/255.0
        
        #load groundtruth data
        shape=utils.load_voxel_data(vox_l_fn).astype(np.int)
        shape_h=utils.load_voxel_data(vox_h_fn).astype(np.int)
       
        #process and reset groundtruth
        shape = sn.binary_dilation(shape)
        shape_h= sn.binary_dilation(shape_h)
        self.target_points = np.argwhere(shape == 1)
        self.target_h_points = np.argwhere(shape_h == 1)
        self.target = shape
        self.target_h = shape_h
        
        return self.ref, self.all_boxes/self.vox_size_l, self.step_vec
        
        
    def initialize_box(self):
        padding = 2
        size = int((self.vox_size_l - padding) / 3.0)
        init_boxes = np.zeros((self.box_num, 6), dtype=np.int)
        count = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    x, y, z = padding + i * size, padding + j * size, padding + k * size
                    x_, y_, z_ = x + size, y + size, z + size
                    init_boxes[count] = [x, y, z, x_, y_, z_]
                    count+=1
        return init_boxes
        
        
    def generate_action_map(self):
        #map between action id and detailed operation parameters
        action_id = 0
        
        #map action id to box_id (27), [c1:x,y,z; c2:x,y,z; delete] (7), range[-2,-1,1,2] (4)
        action_map = np.zeros((27 * 7 * 4, 3), dtype=int)
        map_action = np.zeros((27, 7, 4), dtype=int)
        for i in range(27):
            for j in range(7):
                for k in range(1,5):
                    
                    map_action[i][j][k-1]=action_id
                    
                    if k >= 3:
                        k = k - 5
                    para = np.array([i, j, k])
                    action_map[action_id] = para
                    
                    action_id += 1

        return action_map, map_action
    
    
    def tweak_box(self, action):
        
        try_box=copy.copy(self.all_boxes)
        i, j, k = self.action_map[action]
        
        #delete action
        if j==6:
            midx = try_box[i][0]
            midy = try_box[i][1]
            midz = try_box[i][2]
            try_box[i][0:6] = [midx, midy, midz, midx, midy, midz]
        #edit action
        else:
            try_box[i][j] = try_box[i][j]+ k*self.m_unit

        return try_box
    
    
    def next(self, action):
        
        self.all_boxes=self.tweak_box(action)
        IOU, local_IOU, delete_count= self.compute_increment(self.all_boxes)
        reward=self.compute_reward(IOU, local_IOU, delete_count)
        
        self.last_IOU = IOU
        self.last_delete = delete_count
        self.last_local_IOU = local_IOU        
        
        self.step_count += 1
        self.step_vec=np.zeros((self.max_step), dtype=np.int)
        
        if self.step_count==self.max_step:
            done=True
        else:
            done=False
            self.step_vec[self.step_count]=1
        
        return self.ref, self.all_boxes/self.vox_size_l, self.step_vec, reward, done
    
    
    def next_no_update(self, action):
        
        try_boxes = self.tweak_box(action)
        IOU, local_IOU, delete_count = self.compute_increment(try_boxes)
        reward=self.compute_reward(IOU, local_IOU, delete_count)
                
        step_vec=np.zeros((self.max_step), dtype=np.int)
        if self.step_count+1 >= self.max_step:
            done=True
        else:
            done=False
            step_vec[self.step_count+1]=1
            
        return self.ref, try_boxes/self.vox_size_l, step_vec, reward, done
    
    
    def compute_increment(self, box):
        
        #voxelize the boxes
        canvas = np.zeros_like(self.target, dtype=np.int)
        
        clip_box = np.clip(box, 0, self.vox_size_l-1)
        for i in range(self.box_num):
            [x, y, z, x_, y_, z_] = clip_box[i][0:6]
            canvas[x:x_, y:y_, z:z_] = 1
        
        intersect = canvas & self.target
        i_count = np.sum(intersect == 1)

        union = canvas | self.target
        u_count = np.sum(union == 1)

        delete_count = 0
        sum_single_iou = 0
        
        for i in range(self.box_num):

            if box[i][3]-box[i][0]<=0 or box[i][4]-box[i][1]<=0 or box[i][5]-box[i][2]<=0:
                delete_count+=1

            else:
                single_canvas = np.zeros((self.vox_size_l, self.vox_size_l, self.vox_size_l), dtype=np.int)
                [x, y, z, x_, y_, z_] = clip_box[i][0:6]
                single_canvas[x:x_, y:y_, z:z_] = 1

                single_intersect = single_canvas & self.target
                s_i_count = np.sum(single_intersect == 1)
                
                single_union = single_canvas | self.target
                s_u_count = np.sum(single_union == 1)

                local_iou = float(s_i_count) / float(s_u_count)
                sum_single_iou += local_iou
        
        iou=float(i_count)/float(u_count)
        
        if delete_count==self.box_num:
            local_iou = 0
        else:
            local_iou = sum_single_iou / (self.box_num - delete_count)
            
        return iou, local_iou, delete_count
    
    
    def compute_reward(self, iou, local_iou, delete_count):
        
        r_iou = iou - self.last_IOU
        r_local = local_iou - self.last_local_IOU
        r_parsimony = delete_count - self.last_delete
        
        a=0.1
        b=0.01
        
        reward = r_iou + a*r_local + b*r_parsimony 

        return reward
    
    
    def get_virtual_expert_action(self, valid_mask, random=False):
        
        if not random:
            box_id=self.step_count%self.box_num
        else:
            box_id=np.random.randint(0,self.box_num)
            
        max_action=-1
        max_reward=-1000

        box_level_range=6
        
        #allow delete actions
        if self.step_count>self.max_step*0.5:
            box_level_range=7
        
        for i in range(box_level_range):
            for j in range(4):
                action=self.map_action[box_id,i,j]
                
                if valid_mask[action]==0:
                    continue
                
                s_, boxes_, step_, reward, done = self.next_no_update(action)
                
                if reward>max_reward:
                    max_reward=reward
                    max_action=action       
        
        return max_action        
        
    
    def get_valid_action_mask(self, boxes_normalized):
        
        boxes=boxes_normalized*self.vox_size_l
        valid_mask=np.ones((self.action_num),dtype=np.int)
        
        for a in range(self.action_num):
        
            i, j, k = self.action_map[a]
            box_id=self.step_count%self.box_num
            #only edit an designated box
            if i!=box_id:
                valid_mask[a]=0
            #delete action
            if j==6:
                if boxes[i][3]-boxes[i][0]==0 or boxes[i][4]-boxes[i][1]==0 or boxes[i][5]-boxes[i][2]==0:
                    valid_mask[a]=0
            #edit action
            else:
                bc = boxes[i][j]+ k
                if j<=2:
                    if bc>boxes[i][j+3] or bc < 0:
                        valid_mask[a]=0
                elif j>=3:
                    if bc<boxes[i][j-3] or bc > self.vox_size_l:
                        valid_mask[a]=0
                
        return valid_mask
    
    
    def add_edgeloop_to_one_box(self, c1, c2, subx=2, suby=2, subz=2, v_offset=0, l_offset=0):

        [lx,ly,lz]=c2-c1
        dx=lx/float(subx-1)
        dy=ly/float(suby-1)
        dz=lz/float(subz-1)
        
        v,f=[],[]
        map_index=np.zeros((subx,suby,subz),dtype=np.int)-1
        
        index=0
        #all vertices
        for i in range(subx):
            for j in range(suby):
                for k in range(subz):
                    x=c1[0]+i*dx
                    y=c1[1]+j*dy
                    z=c1[2]+k*dz
                    
                    p=np.array([x,y,z])
                    #only extract boundary-surface points
                    if (p==c1).any() or (p==c2).any():
                        map_index[i,j,k]=index
                        v.append(p)
                        index+=1
        
        for i in range(subx-1):
            for j in range(suby-1):
                for k in [0,subz-1]:
                    v1_i=map_index[i,j,k]
                    v2_i=map_index[i+1,j,k]
                    v3_i=map_index[i+1,j+1,k]
                    v4_i=map_index[i,j+1,k]
                    f.append(np.array([v1_i,v2_i,v3_i,v4_i]))
        
        for i in range(suby-1):
            for j in range(subz-1):
                for k in [0,subx-1]:
                    v1_i=map_index[k,i,j]
                    v2_i=map_index[k,i+1,j]
                    v3_i=map_index[k,i+1,j+1]
                    v4_i=map_index[k,i,j+1]
                    f.append(np.array([v1_i,v2_i,v3_i,v4_i]))

        
        for i in range(subx-1):
            for j in range(subz-1):
                for k in [0,suby-1]:
                    v1_i=map_index[i,k,j]
                    v2_i=map_index[i+1,k,j]
                    v3_i=map_index[i+1,k,j+1]
                    v4_i=map_index[i,k,j+1]
                    f.append(np.array([v1_i,v2_i,v3_i,v4_i]))
                    
        v=np.array(v)
        f=np.array(f)+v_offset
        
        loop_list=[]
        loop_map=np.zeros((v.shape[0],2),np.int)-1
        loop_index=l_offset
        
        lens=[lx,ly,lz]
        maxlen=lens.index(max(lens))
        
        #extract loops
        if maxlen==0:
            for i in range(subx):
                #all indices of this loop
                indices=map_index[i,:,:]
                indices=indices[indices!=-1]+v_offset
                loop_list.append(indices)
                
                #four control points of this loop
                c1=map_index[i,0,0]
                c2=map_index[i,suby-1,subz-1]
                c3=map_index[i,0,subz-1]
                c4=map_index[i,suby-1,0]
                
                #map control points to loop index
                loop_map[c1]=[0,loop_index]
                loop_map[c2]=[0,loop_index]
                loop_map[c3]=[0,loop_index]
                loop_map[c4]=[0,loop_index]
                
                loop_index+=1
        
        if maxlen==1:
            for i in range(suby):
                indices=map_index[:,i,:]
                indices=indices[indices!=-1]+v_offset
                loop_list.append(indices)
                
                c1=map_index[0,i,0]
                c2=map_index[subx-1,i,subz-1]
                c3=map_index[0,i,subz-1]
                c4=map_index[subx-1,i,0]
                
                loop_map[c1]=[1,loop_index]
                loop_map[c2]=[1,loop_index]
                loop_map[c3]=[1,loop_index]
                loop_map[c4]=[1,loop_index]
                
                loop_index+=1
        
        if maxlen==2:
            for i in range(subz):
                indices=map_index[:,:,i]
                indices=indices[indices!=-1]+v_offset
                loop_list.append(indices)
                
                c1=map_index[0,0,i]
                c2=map_index[subx-1,suby-1,i]
                c3=map_index[subx-1,0,i]
                c4=map_index[0,suby-1,i]
                
                loop_map[c1]=[2,loop_index]
                loop_map[c2]=[2,loop_index]
                loop_map[c3]=[2,loop_index]
                loop_map[c4]=[2,loop_index]
                
                loop_index+=1
            
        return v, f, loop_list

        
    def add_edgeloop_to_all(self, boxlist):
        
        #remove degenerated boxes
        newboxlist=np.zeros((boxlist.shape[0],6))
        validcount=0
        for i in range(boxlist.shape[0]):
            if boxlist[i,3]-boxlist[i,0]<=0 or boxlist[i,4]-boxlist[i,1]<=0 or boxlist[i,5]-boxlist[i,2]<=0:
                continue
            newboxlist[validcount]=boxlist[i,:]
            validcount+=1

        #final boxlist
        boxnum=validcount
        boxlist=newboxlist[0:boxnum,:]
        vertexlist = []
        facelist = []
        control_v_list=[]
        
        #store the vertex indices
        loop_list = []
        #map a loop_id into [direction, id, vertex_id1, vertex_id2]
        loop_map = []
        #store the loop ids in each box
        box_loop_list=[]
        
        edge_len=boxlist[:,3:6]-boxlist[:,0:3]      
        max_len=edge_len.max(axis=1)
        max_axis=edge_len.argmax(axis=1)
        
        volume=edge_len[:,0]*edge_len[:,1]*edge_len[:,2]
        sum_volume=np.sum(volume)
        interval=float(sum_volume)/float(self.loop_num)
  
        box_subdiv=np.zeros((boxnum,3),dtype=np.int)
        ctrl_loop_num=0
        
        for i in range(boxnum):
            num=round(volume[i]/interval)
            if i==boxnum-1:
                num=self.loop_num-ctrl_loop_num
            if num<2:
                num=2
            
            sub=np.array([2,2,2])
            sub[max_axis[i]]=num
            box_subdiv[i]=sub

            ctrl_loop_num+=num
            
        offset=0
        loop_offset=0
        
        #record which box a face belongs to
        face_tag=[]
        
        #prepare the mesh structure
        for i in range(boxnum):
            
            v, f, l = self.add_edgeloop_to_one_box(boxlist[i][0:3], boxlist[i][3:6], box_subdiv[i,0], box_subdiv[i,1], box_subdiv[i,2], v_offset=offset, l_offset=loop_offset)

            for j in range(v.shape[0]):
                vertexlist.append(v[j])
            
            for j in range(f.shape[0]):
                facelist.append(f[j,:])
                face_tag.append(i)
            
            #loop_list stores four vertex ids for each loop
            for j in range(len(l)):
                loop_list.append(l[j])
            
            #box_loop_list stores loop_ids for each box
            loops=[]
            for j in range(len(l)):
                loops.append(j+loop_offset)                
            box_loop_list.append(loops)
            
            loop_offset+=len(l)
            offset+=v.shape[0]
        
        vertexlist=np.array(vertexlist)
        facelist=np.array(facelist)
                
        #consolidate the information of each loop
        for i in range(len(loop_list)):
            indices = loop_list[i]
            loop_v = vertexlist[indices]
            c1,c2 = utils.get_corner(loop_v)
            
            #extract control points from v
            control_v_list.append([c1,c2])
            
            #get the loop directions
            if c1[0]==c2[0]:
                direction=0
            elif c1[1]==c2[1]:
                direction=1
            elif c1[2]==c2[2]:
                direction=2
            
            #get control point id
            vid1=0
            vid2=0
            for j in range(len(indices)):
                id=indices[j]
                if (c1==vertexlist[id]).all():
                    vid1=id
                if (c2==vertexlist[id]).all():
                    vid2=id
            
            #map to: direction, loop_id, c1_vertex_id in primitive, c2_vertex_id...
            loop_map.append([direction,i,vid1,vid2])
            
        return vertexlist, facelist, control_v_list, loop_list, loop_map, box_loop_list
    
        
    def output_result(self, log_info, save_path):
        
        #save input shape reference
        self.raw_img.save(save_path + str(self.name) + '.png')
        
        #output file name
        output_name= save_path + log_info + ".off" 
        
        #erode all primitives
        self.cleaned_boxes=copy.copy(self.all_boxes)
        for i in range(self.cleaned_boxes.shape[0]):
            for j in range(3):
                self.cleaned_boxes[i,j+3]=self.all_boxes[i,j+3]-1
        
        #clean all primitives
        self.cleaned_boxes=utils.clean_boxlist_notdel(self.cleaned_boxes)
        
        #handle the case that all primitives are deleted
        if self.cleaned_boxes.shape[0]==0:
            self.cleaned_boxes=np.array([[0,0,0,1,1,1]])
        
        #merge redundant primitives, clean and sort
        if self.step_count==self.max_step:
            self.cleaned_boxes=utils.merge_boxlist(self.cleaned_boxes, self.vox_size_l, self.box_num, 0.85, True)
            self.cleaned_boxes=utils.merge_boxlist(self.cleaned_boxes, self.vox_size_l, self.box_num, 0.9, True)
            self.cleaned_boxes=utils.clean_boxlist(self.cleaned_boxes)
            self.cleaned_boxes=utils.sort_boxlist(self.cleaned_boxes)

        if self.cleaned_boxes.shape[0]!=0:
            #scale the primitives to match the higher resolution of the GT of Mesh-Agent
            self.scaled_box_list = utils.get_scaled_prim_point(self.cleaned_boxes, self.target_points, self.target_h_points)
            
            #avoid overlapping mesh for better visualization
            r_scaled_box_list=self.scaled_box_list+np.random.random(self.scaled_box_list.shape)*0.1
            
            #output result
            utils.save_boxlist(r_scaled_box_list, output_name, self.colors)
    
    
    def save_edgeloop(self, save_path):
        
        vertex_list, face_list, control_v_list, loop_list, loop_map, box_loop_list=self.add_edgeloop_to_all(self.scaled_box_list)
        face_list=face_list+1
        mesh_path=save_path+self.name+'.obj'
        loop_info_path=save_path+self.name+'.loop'
        
        utils.save_mesh_obj(vertex_list, face_list, mesh_path)
        
        with open(loop_info_path, "w") as file:
            file.write(str(len(control_v_list))+" "+str(len(box_loop_list))+"\n")
            file.write('# the coordinates of the loop control points\n')
            for i in range(len(control_v_list)):
                for j in range(2):
                    for k in range(control_v_list[i][j].shape[0]):
                        file.write(str(control_v_list[i][j][k])+" ")
                file.write("\n")
            
            file.write('# the id of the four points of each loop\n')
            for i in range(len(loop_list)):
                for j in range(len(loop_list[i])):
                    file.write(str(loop_list[i][j])+" ")
                file.write("\n")
            
            file.write('# loop direction, loop id, the id of the first control point in the primitive, the id of the second one\n')
            for i in range(len(loop_map)):
                file.write(str(loop_map[i][0])+" "+str(loop_map[i][1])+" "+str(loop_map[i][2])+" "+str(loop_map[i][3])+"\n")
            
            file.write('# loop indices for each box\n')
            for i in range(len(box_loop_list)):
                for j in range(len(box_loop_list[i])):
                    file.write(str(box_loop_list[i][j])+" ")
                file.write("\n")    


