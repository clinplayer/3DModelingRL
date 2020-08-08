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
    parser = argparse.ArgumentParser(description='Prim-Agent')
    
    parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')
    parser.add_argument('--reference', type=str, default='depth', help='type of shape reference, rgb or depth image')
    parser.add_argument('--category', type=str, default='airplane-02691156', help='category name, should be consistent with the name used in file path')
    parser.add_argument('--data_root', type=str, default='../data/', help='root directory of all the data')
    
    parser.add_argument('--load_net', type=str, default='../pretrained/Prim/', help='directory to load the pre-trained network parameters')
    parser.add_argument('--save_result', type=str, default='../data/prim_result/', help='directory to save the modeling results')

    args = parser.parse_args()
    
    return args

def test(agent, env, shape_list):
        
    for shape_count in range(len(shape_list)):
        
        shape_name=shape_list[shape_count]
        vox_l_fn = shape_vox_path+ shape_name+'-16.binvox'
        vox_h_fn = shape_vox_path+ shape_name+'-64.binvox'
        ref_fn = shape_ref_path + shape_name + '.png'
        shape_infopack=[shape_name, vox_l_fn, vox_h_fn, shape_ref_type, ref_fn]
                    
        print('Shape:', shape_count, 'name:', shape_name)
        
        s, box, step = env.reset(shape_infopack)
                    
        while True:
            
            valid_mask = env.get_valid_action_mask(box)
            a = agent.choose_action(s, box, step, valid_mask, 1.0)
            s_, box_, step_, r, done = env.next(a)
                                    
            if done:
                env.output_result(shape_name+'-color', save_result_path)
                env.save_edgeloop(save_result_path)
                break
            
            s = s_
            box = box_    
            step = step_
                    

if __name__ == "__main__":

    args = parse_args()
    
    shape_ref_type=args.reference
    shape_category=args.category
    shape_ref_path=args.data_root + 'shape_reference/'+shape_ref_type+'/'+shape_category + '/'
    shape_vox_path=args.data_root + 'shape_binvox/' + shape_category + '/'
    
    load_net_path=args.load_net + shape_ref_type + '/' + shape_category + '.pth'
    train_shapelist_path=args.data_root + 'shape_list/' + shape_category +'/' + shape_category + '-train.txt'
    test_shapelist_path=args.data_root + 'shape_list/' + shape_category +'/' + shape_category + '-test.txt'
    
    save_result_path = args.save_result + shape_ref_type+'/'+shape_category + '/'
    utils.check_dirs([save_result_path])

    #GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    #shape list
    train_shapelist=utils.load_filelist(train_shapelist_path)
    test_shapelist=utils.load_filelist(test_shapelist_path)
    shapelist=train_shapelist+test_shapelist
    
    #initialize
    env = Environment()
    agent = Agent(load_net_path)
    test(agent, env, shapelist)
    
