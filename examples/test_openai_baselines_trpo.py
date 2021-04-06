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
from baselines.trpo_mpi import trpo_mpi
from baselines.ppo1.mlp_policy import MlpPolicy
import baselines.common.tf_util as U

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def train(env_id, num_timesteps, seed):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()
    #workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    # def policy_fn(name, ob_space, ac_space):
    #     return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
    #         hid_size=32, num_hid_layers=2)

    # Create a new base directory like /tmp/openai-2018-05-21-12-27-22-552435
    # log_dir = os.path.join(energyplus_logbase_dir(), datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
    # if not os.path.exists(log_dir + '/output'):
    #     os.makedirs(log_dir + '/output')
    # os.environ["ENERGYPLUS_LOG"] = log_dir
    # model = os.getenv('ENERGYPLUS_MODEL')
    # if model is None:
    #     print('Environment variable ENERGYPLUS_MODEL is not defined')
    #     os.exit()
    # weather = os.getenv('ENERGYPLUS_WEATHER')
    # if weather is None:
    #     print('Environment variable ENERGYPLUS_WEATHER is not defined')
    #     os.exit()

    # rank = MPI.COMM_WORLD.Get_rank()
    # if rank == 0:
    #     print('train: init logger with dir={}'.format(log_dir)) #XXX
    #     logger.configure(log_dir)
    # else:
    #     logger.configure(format_strs=[])
    #     logger.set_level(logger.DISABLED)

    env = gym.make(env_id)

    trpo_mpi.learn(network='mlp',
                   env=env,
                   total_timesteps=num_timesteps,
                   #timesteps_per_batch=1*1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
                   timesteps_per_batch=16*1024,
                   max_kl=0.01,
                   cg_iters=10,
                   cg_damping=0.1,
                   gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)
    env.close()

if __name__ == '__main__':
	parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--env', '-e', help='environment ID', type=str, default='Eplus-demo-v1')
	parser.add_argument('--seed', '-s', help='RNG seed', type=int, default=0)
	parser.add_argument('--num-timesteps', type=int, default=int(100))
	#parser.add_argument('--save-interval', type=int, default=int(0))
	#parser.add_argument('--model-pickle', help='model pickle', type=str, default='')
	#parser.add_argument('--checkpoint', help='checkpoint file', type=str, default='')
	args = parser.parse_args()
	train(env_id=args.env, num_timesteps=args.num_timesteps, seed=args.seed)