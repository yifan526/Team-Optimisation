import gym
import torch
import numpy as np
import random,os
from runner import Runner
#from smac.env import StarCraft2Env
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args,get_qplex_args
from env.Env5.Env_v3 import MyEnv

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(8)
def get_env(args):
    print(args.env)
    env = MyEnv()

    return env


if __name__ == '__main__':
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    args = get_common_args()
    if args.alg.find('coma') > -1:
        args = get_coma_args(args)
    if args.alg.find('qplex') > -1:
        args = get_mixer_args(args)
        args = get_qplex_args(args)
    elif args.alg.find('central_v') > -1:
        args = get_centralv_args(args)
    elif args.alg.find('reinforce') > -1:
        args = get_reinforce_args(args)
    else:
        args = get_mixer_args(args)
    if args.alg.find('commnet') > -1:
        args = get_commnet_args(args)
    if args.alg.find('g2anet') > -1:
        args = get_g2anet_args(args)
    args.GPU = "cuda:0"# + str(random.randint(0, 5))  # random.randint(0, 5)

    # args.seed = random.randint(0, 1000000)
    setup_seed(args.seed)
    env = get_env(args)
    
    args.n_epoch = 1000
# =============================================================================
#     if 'pacmen' in args.env:
#         args.QPLEX_mixer = 'dmaq'
#         args.n_epoch = 70000
#     elif args.env == '3_vs_2' or args.env == '2_vs_3':
#         args.n_epoch = 150000
#     elif args.env == '3_vs_3_full' or args.env == '3_vs_5_full':
#         args.n_epoch = 600000
#     else:
#         args.n_epoch = 200000
# =============================================================================
    if 'qmix' in args.label or 'qplex' in args.label:
        args.uRNN=False
        args.beta=0
    if 'debug' in args.label or 'qplex' in args.label:
        args.wandb=False
        args.batch_size=8
    print('--------------', args.env, '------------')
    print('label:', args.label, args.GPU, 'seed:', args.seed)

    args.evaluate_cycle = 100
    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.p_state = env_info["p_state"]
    args.episode_limit = env_info["episode_limit"]

    # 神经网络
    runner = Runner(env, args)
    runner.run(0)
    env.close()

    # if not args.evaluate:
    #     runner.run(0)
    # else:
    #     win_rate, _ = runner.evaluate()
    #     print('The win rate of {} is  {}'.format(args.alg, win_rate))
    #     break
    # env.close()
