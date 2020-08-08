import argparse
import numpy as np
import os
from environment import Environment
from network import Net, Agent
import config as p
from utils import utils

shape_ref_type=''
shape_category=''
shape_ref_path=''
shape_vox_path=''
load_net_path=''
save_result_path=''

def parse_args():

    """parse input arguments"""
    parser = argparse.ArgumentParser(description='Mesh-Agent')
    
    parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')
    parser.add_argument('--reference', type=str, default='depth', help='type of shape reference, rgb or depth image')
    parser.add_argument('--category', type=str, default='airplane-02691156', help='category name, should be consistent with the name used in file path')
    parser.add_argument('--data_root', type=str, default='../data/', help='root directory of all the data')
    
    parser.add_argument('--load_net', type=str, default='../pretrained/Mesh/', help='directory to load the pre-trained network parameters')
    parser.add_argument('--save_result', type=str, default='../data/mesh_result/', help='directory to save the modeling results')

    args = parser.parse_args()
    
    return args

def test(agent, env, shape_list):
    
    all_reward=[]
    all_iou_init=[]
    all_iou_edit=[]
    record_reward=[]
    for shape_count in range(len(shape_list)):
        
        shape_name=shape_list[shape_count]
        vox_l_fn = shape_vox_path+ shape_name+'-16.binvox'
        vox_h_fn = shape_vox_path+ shape_name+'-64.binvox'
        prim_mesh_fn = edgeloop_path+shape_name+'.obj'
        loop_info_fn = edgeloop_path+shape_name+'.loop'
        ref_fn = shape_ref_path + shape_name + '.png'
        
        shape_infopack=[shape_name, vox_l_fn, vox_h_fn,  prim_mesh_fn, loop_info_fn, shape_ref_type, ref_fn]
        
        print('Shape:', shape_count, 'name:', shape_name)
        
        valid, s, loop, step = env.reset(shape_infopack)
        if not valid:
            continue
        
        all_iou_init.append(env.last_IOU)
        
        acm_r=0
        
        while True:
            
            valid_mask = env.get_valid_action_mask()
            a = agent.choose_action(s, loop, step, valid_mask, 1.0)
            s_, loop_, step_, r, done = env.next(a)
            acm_r+=r

            if done:
                all_iou_edit.append(env.last_IOU)
                all_reward.append(acm_r) 
                log_info=shape_name+'_r_'+str(format(acm_r, '.4f'))
                
                env.output_result(log_info, save_result_path)
                break
            
            s = s_
            loop = loop_    
            step = step_
    
    return np.mean(all_reward), np.mean(all_iou_init), np.mean(all_iou_edit)

if __name__ == "__main__":
    
    args = parse_args()
    
    shape_ref_type=args.reference
    shape_category=args.category
    shape_ref_path=args.data_root + 'shape_reference/'+shape_ref_type+'/'+shape_category + '/'
    shape_vox_path=args.data_root + 'shape_binvox/' + args.category + '/'
    edgeloop_path=args.data_root + 'prim_result/' +shape_ref_type+'/'+shape_category + '/'
    
    load_net_path=args.load_net + shape_ref_type + '/' + shape_category + '.pth'

    test_shapelist_path=args.data_root + 'shape_list/' + shape_category +'/' + shape_category + '-test.txt'
    
    save_result_path = args.save_result + '/' + shape_category + '/'
    utils.check_dirs([save_result_path])

    #GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    #shape list
    test_shapelist=utils.load_filelist(test_shapelist_path)

    #initialize
    env = Environment()
    agent = Agent(load_net_path)
    mean_reward, mean_IOU_init, mean_IOU_edit = test(agent, env, test_shapelist)
    
    print('mean reward:', mean_reward)
    print('mean IOU init:', mean_IOU_init)
    print('mean IOU edit:', mean_IOU_edit)

    


