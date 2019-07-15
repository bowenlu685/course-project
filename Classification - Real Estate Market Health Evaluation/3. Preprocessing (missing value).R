##missing value
setwd("/Users/haocheng/Desktop/zillow")
market_health <- read.csv("DataMerged.csv")
str(market_health)
head(market_health)
summary(market_health)

install.packages("VIM")
library(VIM)

?kNN()

mydata <- kNN(market_health, k=100)
summary(mydata)

mydata <- subset(mydata, select = X:Negative.Equity.Deliquency.Rate...)
summary(mydata)

head(mydata)


write.csv(mydata, file = "Final_data.csv")

