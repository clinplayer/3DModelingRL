import sys
sys.path.append('../Mesh-Agent')
import os
from environment import Environment
from network import Net, Agent
from utils import utils
sys.path.append('../demo')


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#initialize
env = Environment()
agent = Agent('eval/mesh.pth')

#save path
save_path='mesh_result/'
utils.check_dirs([save_path])

shape_infopack=['demo', 'eval/demo-16.binvox', 'eval/demo-64.binvox', 'prim_result/demo.obj', 'prim_result/demo.loop', 'rgb', 'demo.png']
valid, s, box, step = env.reset(shape_infopack)
acm_r=0

step_interval=10

while True:

    valid_mask = env.get_valid_action_mask()
    a = agent.choose_action(s, box, step, valid_mask, 1.0)
    s_, box_, step_, r, done = env.next(a)
                
    acm_r+=r
    if env.step_count%step_interval==0:
        log_info='demo_step_'+str(env.step_count)
        print(log_info)
        env.output_result(log_info, save_path)

    if done:
        log_info='demo_finish_reward_'+str(acm_r)
        print(log_info)
        env.output_result(log_info, save_path)
        break

    s = s_
    box = box_    
    step = step_
                
        

