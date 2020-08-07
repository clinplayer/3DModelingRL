import numpy as np
import config as p

class Memory():
    
    def __init__(self, capacity):
        
        super(Memory, self).__init__()
                
        self.ref_size = p.REF_IMG_RESO
        self.box_num = p.BOX_NUM
        self.capacity = capacity
        self.max_step=p.MAX_STEP
        
        self.s_mem=np.zeros((self.capacity, 1, self.ref_size, self.ref_size))
        self.obj_mem=np.zeros((self.capacity, self.box_num, 6))
        self.step_mem=np.zeros((self.capacity, self.max_step))
        
        self.a_mem=np.zeros((self.capacity, 1))
        self.r_mem=np.zeros((self.capacity, 1))
        
        self._s_mem=np.zeros((self.capacity, 1, self.ref_size, self.ref_size))
        self._obj_mem=np.zeros((self.capacity, self.box_num, 6))
        self._step_mem=np.zeros((self.capacity, self.max_step))
        
        self.memory_counter=0
    
    
    def store(self, s, obj, step, a, r, s_, obj_, step_):

        index = self.memory_counter % self.capacity  
        
        self.s_mem[index, :] = s
        self.obj_mem[index,:]= obj
        self.step_mem[index, :]=step
        
        self.a_mem[index, :] = a
        self.r_mem[index, :] = r
        
        self._s_mem[index, :]= s_
        self._obj_mem[index,:]= obj_
        self._step_mem[index, :]=step_

        self.memory_counter += 1
        
    
    def clear(self):
        self.memory_counter=0
        
    
    def sample(self,num):
        
        if self.memory_counter<self.capacity:
            indices = np.random.choice(self.memory_counter, size=num)
        else:
            indices = np.random.choice(self.capacity, size=num)
        
        bs = self.s_mem[indices, :]
        bobj = self.obj_mem[indices, :]
        bstep = self.step_mem[indices, :]

        ba = self.a_mem[indices, :]
        br = self.r_mem[indices, :]
        
        bs_ = self._s_mem[indices, :]
        bobj_ = self._obj_mem[indices, :]
        bstep_ = self._step_mem[indices, :]
        
        return bs, bobj, bstep, ba, br, bs_, bobj_, bstep_
    
    

            
            
    