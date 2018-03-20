#import quantmod

library(quantmod)
library(data.table)
library(dplyr)

dt_GBPJPY <-fread("GBPJPY.csv", header=F)
colnames(dt_GBPJPY) <- c("Date","Time","Open","High","Low","Close","Volume")
#範囲指定
class(dt_GBPJPY$Date)
dt_GBPJPY2 <- dt_GBPJPY[dt_GBPJPY$Date >= "2017.02.00" & dt_GBPJPY$Date< "2017.03.00" ,]
write.csv(dt_GBPJPY2, "GBPJPY2.csv", quote=F,row.names=F)


cn=c("Date","Time","Open","High","Low","Close","Volume") #MT4のcsvのラベル

data <- read.zoo("GBPJPY2.csv",sep=",",header=T,index=1:2,tz="",format="%Y.%m.%d%H:%M") #

candleChart(data,subset='2017-02-01 19:10:00::2017-02-01') #指定の範囲でローソク足チャート表示
dev.off()        

plot(sin, -pi, 2*pi)                                   # plot(関数名, 下限, 上限)
gauss.density <- function(x) 1/sqrt(2*pi)*exp(-x^2/2)  # 標準正規分布の密度
plot(gauss.density,-3,3)