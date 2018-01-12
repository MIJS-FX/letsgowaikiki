# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 17:26:11 2017

@author: Takeshi_Umezaki
"""

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym
import numpy as np

env = gym.make("CartPole-v0")
print("observation space   : {}".format(env.observation_space))
print("action space        : {}".format(env.action_space))
obs = env.reset() #初期化
#env.render()#レンダリングした環境を見せてくれる
print("initial observation : {}".format(obs))

action = env.action_space.sample()
obs, r, done, info = env.step(action)

### どんな値が入っているのか確認！
print('next observation    : {}'.format(obs))
print('reward              : {}'.format(r))
print('done                : {}'.format(done))
print('info                : {}'.format(info))

env.render()

class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels=50):
        #super(QFunction, self).__init__(##python2.x用
        super().__init__(#python3.x用
            l0=L.Linear(obs_size, n_hidden_channels),
            l1=L.Linear(n_hidden_channels,n_hidden_channels),
            l2=L.Linear(n_hidden_channels, n_actions))
        
    def __call__(self, x, test=False): 
        """
        x ; 観測#ここの観測って、stateとaction両方？
        test : テストモードかどうかのフラグ
        """
        h = F.tanh(self.l0(x)) #活性化関数は自分で書くの？
        h = F.tanh(self.l1(h))
        return chainerrl.action_value.DiscreteActionValue(self.l2(h))

obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n
q_func = QFunction(obs_size, n_actions)
#q_func.to_gpu(0) ## GPUを使いたい人はこのコメントを外す

optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func) #設計したq関数の最適化にAdamを使う
gamma = 0.95
explorer = chainerrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.action_space.sample)
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity = 10**6)
phi = lambda x:x.astype(np.float32, copy=False)##型の変換(chainerはfloat32型。float64は駄目)

agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=500,     phi=phi)

import time
n_episodes = 200
max_episode_len = 200
start = time.time()
for i in range(1, n_episodes + 1):
    obs = env.reset()
    reward = 0
    done = False
    R = 0  # return (sum of rewards)
    t = 0  # time step
    while not done and t < max_episode_len:
        # 動きを見たければここのコメントを外す
        # env.render()
        action = agent.act_and_train(obs, reward)
        obs, reward, done, _ = env.step(action)
        R += reward
        t += 1
    if i % 10 == 0:
        print('episode:', i,
              'R:', R,
              'statistics:', agent.get_statistics())
    agent.stop_episode_and_train(obs, reward, done)
print('Finished, elapsed time : {}'.format(time.time()-start))

for i in range(10):
    obs = env.reset()
    done = False
    R = 0
    t = 0
    while not done and t < 200:
        # env.render()
        action = agent.act(obs)
        obs, r, done, _ = env.step(action)
        R += r
        t += 1
    print('test episode:', i, 'R:', R)
    agent.stop_episode()
    
chainerrl.experiments.train_agent_with_evaluation(
    agent, env,
    steps=2000,           # 2000step学習
    eval_n_runs=10,       #  評価(テスト)を10回する
    max_episode_len=200,  # それぞれの評価に対する長さの最大(200s)
    #eval_frequency=1000,  # テストを1000stepの学習ごとに実施
    eval_interval=1,
    outdir='result')      # 'result'フォルダに保存

