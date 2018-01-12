import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np
import pandas as pd

class FxBoard(): #todo
    glaph = np.array([0] * 9, dtype=np.float32)
    fee = 110
    position = 0
    profit = 0
    end_value = 0
    def reset(self):
        FxBoard.profit = 0 #利益
        FxBoard.fee = 0.1 #販売手数料
        FxBoard.position = 0 # -1 売り 0 なし +1 買い
    def settlement(self): # 決済
        FxBoard.profit = 0 #利益計算
        if(FxBoard.position == 1):
            FxBoard.profit = FxBoard.end_value - FxBoard.glaph[8] - FxBoard.fee
        if(FxBoard.position == 0):
            FxBoard.profit = 0
        if(FxBoard.position == -1):
            FxBoard.profit = FxBoard.glaph[8] - FxBoard.end_value - FxBoard.fee

#ゲームボード
class Board():
    def reset(self):
        self.board = np.array([0] * 9, dtype=np.float32)
        self.winner = None
        self.missed = False
        self.done = False

    def move(self, act, turn):
        if self.board[act] == 0:
            self.board[act] = turn
            self.check_winner()
        else:
            self.winner = turn*-1
            self.missed = True
            self.done = True

    def check_winner(self):
        win_conditions = ((0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6))
        for cond in win_conditions:
            if self.board[cond[0]] == self.board[cond[1]] == self.board[cond[2]]:
                if self.board[cond[0]]!=0:
                    self.winner=self.board[cond[0]]
                    self.done = True
                    return
        if np.count_nonzero(self.board) == 9:
            self.winner = 0
            self.done = True

    def get_empty_pos(self):
        empties = np.where(self.board==0)[0]
        if len(empties) > 0:
            return np.random.choice(empties)
        else:
            return 0

    def show(self):
        row = " {} | {} | {} "
        hr = "\n-----------\n"
        tempboard = []
        for i in self.board:
            if i == 1:
                tempboard.append("○")
            elif i == -1:
                tempboard.append("×")
            else:
                tempboard.append(" ")
        print((row + hr + row + hr + row).format(*tempboard))

#explorer用のランダム関数オブジェクト
class RandomActor:
    def __init__(self, board):
        self.board = board
        self.random_count = 0
    def random_action_func(self):
        self.random_count += 1
        return self.board.get_empty_pos()

#explorer用のランダム関数オブジェクト
class RandomActorFx:
    def __init__(self):
        self.random_count = 0
        self.baibai = [-1,0,1]
    def random_action_func(self):
        #return np.random.choice(-1,0,1)
        self.random_count += 1
        return np.random.choice(self.baibai)
#Q関数
class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels=81):
        super().__init__(
            l0=L.Linear(obs_size, n_hidden_channels),
            l1=L.Linear(n_hidden_channels, n_hidden_channels),
            l2=L.Linear(n_hidden_channels, n_hidden_channels),
            l3=L.Linear(n_hidden_channels, n_actions))
    def __call__(self, x, test=False):
        #-1を扱うのでleaky_reluとした
        h = F.leaky_relu(self.l0(x))
        h = F.leaky_relu(self.l1(h))
        h = F.leaky_relu(self.l2(h))
        return chainerrl.action_value.DiscreteActionValue(self.l3(h))

class QFunctionFx(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels=81):
        super().__init__(
            l0=L.Linear(obs_size, n_hidden_channels),
            l1=L.Linear(n_hidden_channels, n_hidden_channels),
            l2=L.Linear(n_hidden_channels, n_hidden_channels),
            l3=L.Linear(n_hidden_channels, n_actions))
    def __call__(self, x, test=False):
        #-1を扱うのでleaky_reluとした
        h = F.leaky_relu(self.l0(x))
        h = F.leaky_relu(self.l1(h))
        h = F.leaky_relu(self.l2(h))
        return chainerrl.action_value.DiscreteActionValue(self.l3(h))


df= pd.read_csv('GBPJPY_15_201703.csv')

fxb = FxBoard()

fxb.reset()
print(df.iloc[1,3])
npdata1 = np.zeros(9)
#npdata1 = df.iloc[1:9,3].T # TODO 改行が入ってしまう
npdata1[0] = df.iloc[1,3]
npdata1[1] = df.iloc[2,3]
npdata1[2] = df.iloc[3,3]
npdata1[3] = df.iloc[4,3]
npdata1[4] = df.iloc[5,3]
npdata1[5] = df.iloc[6,3]
npdata1[6] = df.iloc[7,3]
npdata1[7] = df.iloc[8,3]
npdata1[8] = df.iloc[9,3]

FxBoard.end_value = df.iloc[10,3] # 終値
#fxb.glaph = npdata1.astype(np.float64)
FxBoard.glaph = npdata1

#fxb.setglaph(df[1])
#fxb.setglaph(fxb.glaph)

# ボードの準備 →todo この部分をFxの環境にする
#b = Board()
# explorer用のランダム関数オブジェクトの準備

#ra = RandomActor(b)
rafx = RandomActorFx()

# 環境と行動の次元数
obs_size = 9 #学習させるパラメータ数。今回は9個（時系列の数）
n_actions = 3 # 売り・ステイ・買い
# Q-functionとオプティマイザーのセットアップ
q_func = QFunction(obs_size, n_actions)
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)
# 報酬の割引率
gamma = 0.95
# Epsilon-greedyを使ってたまに冒険。50000ステップでend_epsilonとなる
explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
    start_epsilon=1.0, end_epsilon=0.3, decay_steps=50000, random_action_func=rafx.random_action_func)
# Experience ReplayというDQNで用いる学習手法で使うバッファ
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
# Agentの生成（replay_buffer等を共有する2つ）
phi = lambda x:x.astype(np.float32, copy=False)##型の変換(chainerはfloat32型。float64は駄目)


agent_fx = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=500,     phi=phi)

#学習ゲーム回数
#n_episodes = 20000
n_episodes = 2
#カウンタの宣言
miss = 0
win = 0
draw = 0

#エピソードの繰り返し実行
for i in range(1, n_episodes + 1):
    #b.reset()
    reward = 0 #報酬
    #agents = [agent_fx]
    #turn = np.random.choice([0, 1])
    last_state = None

    line = 0
    # グラフの最初から終わりまで
    print(len(df))
    
    print(line >= len(df)-10)
    
    while line <= len(df)-10 :
        #配置マス取得
        #action = agents[turn].act_and_train(b.board.copy(), reward)
        # Todo!!
        #action = agent_fx.act_and_train(b.board.copy(), reward)
        print(FxBoard.glaph.copy())
        #****************************************** TODO
        action = agent_fx.act_and_train(FxBoard.glaph.copy(), reward)
        print(action) #売買。0,1,2のはず。
        #配置を実行
        #b.move(action, 1)
        #売買を実行
        FxBoard.position = action
        
        #報酬の算出
        fxb.settlement()
        reward = FxBoard.profit

        #エピソードを終了して学習
        agent_fx.stop_episode_and_train(FxBoard.glaph.copy(), reward, True)

    #コンソールに進捗表示
    if i % 100 == 0:
        print("episode:", i, " / rnd:", rafx.random_count, " / epsilon:", agent_fx.explorer.epsilon)
        rafx.random_count = 0
    if i % 10000 == 0:
        # 10000エピソードごとにモデルを保存
        agent_fx.save("result_" + str(i))

#$agent_fx.load("resultfx_20000")  #←これを追加
print("Training finished.")

#人間のプレーヤー
class HumanPlayer:
    def act(self, board):
        valid = False
        while not valid:
            try:
                act = input("Please enter 1-9: ")
                act = int(act)
                if act >= 1 and act <= 9 and board[act-1] == 0:
                    valid = True
                    return act-1
                else:
                    print("Invalid move")
            except Exception as e:
                print(act +  " is invalid")

#検証
human_player = HumanPlayer()
for i in range(10):
    b.reset()
    dqn_first = np.random.choice([True, False])
    while not b.done:
        #DQN
        if dqn_first or np.count_nonzero(b.board) > 0:
            b.show()
            action = agent_p1.act(b.board.copy())
            b.move(action, 1)
            if b.done == True:
                if b.winner == 1:
                    print("DQN Win")
                elif b.winner == 0:
                    print("Draw")
                else:
                    print("DQN Missed")
                agent_p1.stop_episode()
                continue
        #人間
        b.show()
        action = human_player.act(b.board.copy())
        b.move(action, -1)
        if b.done == True:
            if b.winner == -1:
                print("HUMAN Win")
            elif b.winner == 0:
                print("Draw")
            agent_p1.stop_episode()

print("Test finished.")
