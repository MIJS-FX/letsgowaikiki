import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np
import pandas as pd

class FxBoard(): #todo
    def __init__(self):
        self.glaph = np.array([0] * 9, dtype=np.float32)
        self.fee = 0.01
        self.position = 0
        self.profit = 0
        self.end_value = 0
    def reset(self):
        self.profit = 0 #利益
        self.fee = 0.01 #販売手数料
        self.position = 0 # -1 売り 0 なし +1 買い
    def settlement(self): # 決済
        self.profit = 0 #利益計算
        if(self.position == 1):
            self.profit = self.glaph[8] - self.end_value - self.fee
        if(self.position == 2):
            self.profit = 0
        if(self.position == 3):
            self.profit = self.end_value - self.glaph[8] - self.fee

#explorer用のランダム関数オブジェクト
class RandomActorFx:
    def __init__(self):
        self.random_count = 0
        self.baibai = [1,2,3]
    def random_action_func(self):
        #return np.random.choice(-1,0,1)
        self.random_count += 1
        return np.random.choice(self.baibai)
#Q関数
class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels=27):
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

def setgraph(df):
    npdata1 = np.zeros(9)
    npdata1[0] = df.iloc[1,3]
    npdata1[1] = df.iloc[2,3]
    npdata1[2] = df.iloc[3,3]
    npdata1[3] = df.iloc[4,3]
    npdata1[4] = df.iloc[5,3]
    npdata1[5] = df.iloc[6,3]
    npdata1[6] = df.iloc[7,3]
    npdata1[7] = df.iloc[8,3]
    npdata1[8] = df.iloc[9,3]
    return npdata1

df= pd.read_csv('GBPJPY_15_201703_img.csv')

# Fx環境初期化
fxb = FxBoard()
#fxb.reset()

# explorer用のランダム関数オブジェクトの準備
rafx = RandomActorFx()

# 環境と行動の次元数
obs_size = 9 #学習させるパラメータ数。今回は9個（時系列の数）
n_actions = 3 # 売り・ステイ・買い
# Q-functionとオプティマイザーのセットアップ #TODO
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

#optimizer 何を使って最適化するか。chainerにいろいろと組み込まれている。optimizersリストはこちら
#gamma 報酬の割引率.過去の結果をどのくらい重要視するか
#explorer 次の戦略を考えるときの方法
#replay_buffer Experience Replayを実行するかどうか

agent_fx = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=500,     phi=phi)

#学習ゲーム回数
n_episodes = 1
#n_episodes = 2
n_exec = 0

#エピソードの繰り返し実行
for i in range(1, n_episodes + 1):
    reward = 0 #報酬
    last_state = None

    line = 1
    str_debug = ""
    #while line <= len(df)-10 :
    while line <= 30 : # for debug
        fxb.reset()
        fxb.glaph = setgraph(df[line:line + 10])
        fxb.end_value = df.iloc[line + 10,3] # 終値

        #売買判断取得
        action = agent_fx.act_and_train(fxb.glaph.copy(), reward)
        n_exec = n_exec + 1
        #売買を実行
        fxb.position = action

        #報酬の算出
        fxb.settlement()
        reward = fxb.profit

        #エピソードを終了して学習
        agent_fx.stop_episode_and_train(fxb.glaph.copy(), reward, True)
        # For debug 報酬とアクションを文字列として表示
        str_debug = str_debug + str(action)+ ":" + str(reward) + ","
        
        line = line + 1

    #コンソールに進捗表示
    #print("episode:", i)
    if i % 1 == 0:
        print("episode:", i, " / rnd:", rafx.random_count,"/",n_exec, " / epsilon:", agent_fx.explorer.epsilon)
        print(str_debug)
        rafx.random_count = 0
        n_exec = 0
    if i % 1 == 0:
        # 500エピソードごとにモデルを保存
        agent_fx.save("resultfx_" + str(i))

print("Training finished.")

#agent_fx.load("resultfx_2000")  #←これを追加

#検証
line=1
print("教育に含まれるデータ")
while line <= 30 : 
    fxb.reset()
    fxb.glaph = setgraph(df[line:line+10])
    fxb.end_value = df.iloc[line+10,3] # 終値

    #売買判断取得
    action = agent_fx.act(fxb.glaph.copy())

    #報酬の算出
    fxb.position = action
    fxb.settlement()
    reward = fxb.profit
    
    agent_fx.stop_episode()
    
    print("  line:",line,"  売買:",action,"　利益:",reward)
    
    line = line + 1

print("教育に含まないデータ")
line=100

while line <= 130 : 
    fxb.glaph = setgraph(df[line:line+10])
    fxb.end_value = df.iloc[line+10,3] # 終値

    #売買判断取得
    action = agent_fx.act(fxb.glaph.copy())

    #報酬の算出
    fxb.position = action
    fxb.settlement()
    reward = fxb.profit
    
    agent_fx.stop_episode()
    print("  line:",line,"  売買:",action,"　利益:",reward)
    
    line = line + 1

'''
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
'''