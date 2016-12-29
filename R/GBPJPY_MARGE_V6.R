# 12/29 時系列を１つにし、３０項目とする。３０バージョン 平均化
setwd("C:\\Users\\sisco\\Documents\\GitHub\\letsgowaikiki\\data")             # 作業ディレクトリを変更する
getwd()                     # 現在の作業ディレクトリを確認する# 複数のデータファイルを一括してリストに読み込む

dt_all <- data.frame()

# 読み込んだcsvをマージする
dt_ashi15 <- read.table("GBPJPY15.csv", header=F, sep=",")

# 平均化
ashi15Aves <- colMeans(dt_ashi15[,3,drop=FALSE])
dt_ashi15[,c(3,4,5,6)] <-  dt_ashi15[,c(3,4,5,6)] - ashi15Aves

nrow_ashi15   <- nrow(dt_ashi15)-31 #最後の31行（項目が30個なので）まで
i15 <- 0
# # 空行テーブル


for(i15 in 1:nrow_ashi15){
   i15 <- i15 + 1
   i15e <- i15+30
   str_date <- as.character(dt_ashi15[i15,1]) #日
   str_time <- as.character(dt_ashi15[i15,2]) #時
   dt_endprice <- t(dt_ashi15[c(i15:i15e),6]) #終値
   
   dt_record <- data.frame(cbind(str_date,str_time,dt_endprice))
   dt_all <- rbind(dt_all,dt_record)
}
# write.dt(dt_all, "output.txt", quote=F,col.names=F, append=F,row.names=F)
write.csv(dt_all, "GBPJPYALL.csv", quote=F,row.names=F)

