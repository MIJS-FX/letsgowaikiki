#import quantmod

library(quantmod)
library(data.table)
library(dplyr)
library("magick")

cn=c("Date","Time","Open","High","Low","Close","Volume") #MT4のcsvのラベル

data <- read.zoo("GBPJPY_15_201703.csv",sep=",",header=F,index=1:2,tz="",format="%Y.%m.%d%H:%M",col.names=cn) #


# #subseted_data <- data[1:20]
# par(xaxt="n")
# par(yaxt="n")
# #plt = c(0, 1, 0, 1)
# #par("usr")
# #mai = c(0, 0, 0, 0)
# candleChart(subseted_data)
# #chartSeries(subseted_data)

# jpeg保存にする

for(i in 1:100){
  first_line <- i
  last_line <- i + 20
  subseted_data <- data[first_line:last_line]
  jpeg("candle.jpg", width=200, height=200)
  candleChart(subseted_data)
  dev.off()
  # 日時ファイル名。断念
  #f_name = candle + print(subseted_data[1,1])
  f_name <- paste("candle",sprintf("%05d",i),".jpg",sep = "")
  
  #dev.copy(jpeg,"candle.jpg",width=200,height=200)
  #dev.off()
  cnd_img <- image_read("candle.jpg")
  cnd_img_s <- image_crop(image = cnd_img, geometry = "103x123+51+29")
  image_write(cnd_img_s, path = f_name, format = "jpg")
}
dev.off()
