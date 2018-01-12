import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np
import pandas as pd
class FxBoard(): #todo
    glaph = np.array([0] * 9, dtype=np.float32)
    def reset(self):
        self.profit = 0 #利益
        self.fee = 110 #販売手数料
        self.position = 0 # -1 売り 0 なし +1 買い
    def settlement(self): # 決済
        self.profit = 0 #利益計算
        if(self.position == 1):
            self.profit = self.glaph[9] - self.glaph[8] - self.fee
            #self.profit = self.pd11[1,9] - self.pd11[1,8] - self.fee
        if(self.position == -1):
            self.profit = self.glaph[8] - self.glaph[9] - self.fee
            #self.profit = self.pd11[1,8] - self.pd11[1,9] - self.fee
    def setglaph(self,pd11): # 決済
        self.glaph[1] = pd11[1,1]

df= pd.read_csv('GBPJPY_15_201703.csv')

fxb = FxBoard()
print(df.iloc[1,3])

