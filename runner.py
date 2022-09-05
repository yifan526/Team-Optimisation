import numpy as np
import os,csv
from common.rollout import RolloutWorker, CommRolloutWorker
from agent.agent import Agents, CommAgents
from common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
from tqdm import tqdm

class Runner:
    def __init__(self, env, args):
        self.env = env
        self.csv_dir = f'./csv_file/{args.env}'
        self.csv_path = f'{self.csv_dir}/seed_{args.seed}_{args.label}.csv'


        self.win_dir = f'{self.csv_dir}/win_rate/{args.label}'
        self.win_path = f'{self.win_dir}/seed_{args.seed}_{args.label}.csv'
        if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:  # communication agent
            self.agents = CommAgents(args)
            self.rolloutWorker = CommRolloutWorker(env, self.agents, args)
        else:  # no communication agent
            self.agents = Agents(args)
            self.rolloutWorker = RolloutWorker(env, self.agents, args)

        if not args.evaluate and args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find('reinforce') == -1:  # these 3 algorithms are on-poliy
            self.buffer = ReplayBuffer(args)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []
        self.save_path = self.args.result_dir + f'/{args.env}/{args.label}/{args.seed}'

        for item_file in [self.save_path, self.csv_dir]:
            if not os.path.exists(item_file):
                os.makedirs(item_file)
        if not os.path.exists(self.win_dir):
            os.makedirs(self.win_dir)
        # 用来保存plt和pkl
        self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map


    def run(self, num):
        episode_rewards = []
        time_steps, train_steps, evaluate_steps = 0, 0, -1
        for epoch in tqdm(range(self.args.n_epoch)):  # 20 0000

            episodes = []
            # 收集self.args.n_episodes个episodes
            for episode_idx in range(self.args.n_episodes):
                episode, _, _, steps = self.rolloutWorker.generate_episode(episode_idx)
                episodes.append(episode)
                time_steps += steps
                # print(_)
            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find('reinforce') > -1:
                self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon,time_steps)
                train_steps += 1
            else:
                self.buffer.store_episode(episode_batch)
                if self.buffer.can_sample(self.args.batch_size):
                    for train_step in range(self.args.train_steps):
                        mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                        self.agents.train(mini_batch, train_steps,time_steps=time_steps)
                        train_steps += 1
            if epoch % self.args.evaluate_cycle == self.args.evaluate_cycle - 1:
                win_rate, episode_reward = self.evaluate()
                episode_rewards.append(episode_reward)
                self.writereward(self.csv_path, episode_reward, time_steps)
                self.writereward(self.win_path, win_rate,time_steps)
                # print('win_rate is ', win_rate)
                self.win_rates.append(win_rate)
                self.episode_rewards.append(episode_reward)


    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag, _ = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward

            win_number += win_tag
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch
    def writereward(self, path, reward, step):
        if os.path.isfile(path):
            with open(path, 'a+') as f:
                csv_write = csv.writer(f)
                csv_write.writerow([step, reward])
        else:
            with open(path, 'w') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(['step', 'reward'])
                csv_write.writerow([step, reward])









