my_data <- read.csv(file.choose())
install.packages("dplyr")
View(my_data)

y <- my_data$MarketHealthIndex
x1 <- my_data$SellForGain
x2 <- my_data$ZHVI
x3 <- my_data$MoM
x4 <- my_data$YoY
x5 <- my_data$ForecastYoYPctChange
x6 <- my_data$NegativeEquity
x7 <- my_data$Delinquency
x8 <- my_data$DaysOnMarket
x9 <- my_data$Zri
x10 <- my_data$QoQ
x11 <- my_data$BEPropCount
x12 <- my_data$SampleRate
x13 <- my_data$MedBE
x14 <- my_data$MedPR
x15 <- my_data$TotalAmountofNegativeEquityMillions
x16 <- my_data$TotalNumberofHomesinNegativeEquity
x17 <- my_data$NegativeEquityDeliquencyRate
# MANOVA test
res <- manova(cbind(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17) ~ y, data = my_data)
summary(res)

