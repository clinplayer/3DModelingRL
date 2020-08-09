import sys
sys.path.append('../Prim-Agent')
import os
from environment import Environment
from network import Net, Agent
from utils import utils
sys.path.append('../demo')


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#initialize
env = Environment()
agent = Agent('eval/prim.pth')

#save path
save_path='prim_result/'
utils.check_dirs([save_path])

shape_infopack=['demo', 'eval/demo-16.binvox', 'eval/demo-64.binvox', 'rgb', 'demo.png']
s, box, step = env.reset(shape_infopack)
acm_r=0

step_interval=20

while True:

    valid_mask = env.get_valid_action_mask(box)
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
        env.save_edgeloop(save_path)
        break

    s = s_
    box = box_    
    step = step_
                
        

