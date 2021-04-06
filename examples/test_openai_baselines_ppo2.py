import sys
import re
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np
import argparse

from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module

import energym
from baselines.ppo2 import ppo2
from baselines.ppo1.mlp_policy import MlpPolicy
import baselines.common.tf_util as U

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def get_learn_function(alg):
    try:
        alg_module = import_module(name=alg,package="baselines."+alg)
    except ImportError as err:
        print('Error:', err)
    return alg_module.learn


def train(env_id, alg, num_timesteps, seed):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    env = gym.make(env_id)
    #learn=get_learn_function(alg)

    ppo2.learn(network='mlp',
            env=env,
            total_timesteps=num_timesteps,
            #timesteps_per_batch=1*1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
            #timesteps_per_batch=16*1024,
            #max_kl=0.01,
            #cg_iters=10,
            #cg_damping=0.1,
            #gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3
            )
    env.close()

if __name__ == '__main__':
    parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', '-e', help='environment ID', type=str, default='Eplus-demo-v1')
    parser.add_argument('--alg', '-a', help='Learn DRL algorithm', type=str, default='trpo_mpi')
    parser.add_argument('--seed', '-s', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(100))
    #parser.add_argument('--save-interval', type=int, default=int(0))
    #parser.add_argument('--model-pickle', help='model pickle', type=str, default='')
    #parser.add_argument('--checkpoint', help='checkpoint file', type=str, default='')
    args = parser.parse_args()
    train(env_id=args.env, alg=args.alg, num_timesteps=args.num_timesteps, seed=args.seed)