library(quantmod)
library(data.table)
library(dplyr)
library(magick)

cn=c("Date","Time","Open","High","Low","Close","Volume") #MT4のcsvのラベル

data <- read.zoo("GBPJPY_15_201703.csv",sep=",",header=F,index=1:2,tz="",format="%Y.%m.%d%H:%M",col.names=cn) #
i <- 0
for(i in 1:20){
  first_line <- i
  last_line <- i + 20
  subseted_data <- data[first_line:last_line]
  jpeg("candle.jpg", width=200, height=200)
  candleChart(subseted_data)
  dev.off()
  # 日時ファイル名。断念
  f_name <- paste("candle",sprintf("%05d",i),".jpg",sep = "")
  
  cnd_img <- image_read("candle.jpg")
  cnd_img_s <- image_crop(image = cnd_img, geometry = "103x123+51+29")
  image_write(cnd_img_s, path = f_name, format = "jpg")
}
dev.off()

