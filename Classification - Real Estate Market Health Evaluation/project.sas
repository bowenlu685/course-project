*variable selection;
proc import datafile = "C:/Users/a/Desktop/BIA-652/Project/after_normalization_round.csv" 
 out=work.project dbms=csv replace;
 run;

proc phreg data=project outest=project1;
 model MarketHealthIndex = SellForGain ZHVI MoM YoY ForecastYoYPctChange NegativeEquity Delinquency DaysOnMarket Zri QoQ BEPropCount SampleRate MedBE MedPR TotalAmountofNE TotalNumberofHomesinNE NEDeliquencyRate /
 selection=forward slentry=0.15 details;
run;
proc phreg data=project outest=project2;
 model MarketHealthIndex = SellForGain ZHVI MoM YoY ForecastYoYPctChange NegativeEquity Delinquency DaysOnMarket Zri QoQ BEPropCount SampleRate MedBE MedPR TotalAmountofNE TotalNumberofHomesinNE NEDeliquencyRate /
 selection=backward slstay=0.30 details;
run;
proc phreg data=project outest=project3;
 model MarketHealthIndex = SellForGain ZHVI MoM YoY ForecastYoYPctChange NegativeEquity Delinquency DaysOnMarket Zri QoQ BEPropCount SampleRate MedBE MedPR TotalAmountofNE TotalNumberofHomesinNE NEDeliquencyRate /
 selection=stepwise slentry=0.15 slstay=0.30 details;
run;
proc reg data=project outest=project4;
 model MarketHealthIndex = SellForGain ZHVI MoM YoY ForecastYoYPctChange NegativeEquity Delinquency DaysOnMarket Zri QoQ BEPropCount SampleRate MedBE MedPR TotalAmountofNE TotalNumberofHomesinNE NEDeliquencyRate /
 selection=rsquare adjrsq cp aic details;
run;


*multiple regression analysis (before the variable selection);
proc reg data=project Rsquare;
 model MarketHealthIndex = SellForGain ZHVI MoM YoY ForecastYoYPctChange NegativeEquity Delinquency DaysOnMarket Zri QoQ BEPropCount SampleRate MedBE MedPR TotalAmountofNE TotalNumberofHomesinNE NEDeliquencyRate / stb corrb r p;
 run;
ods graphics on;
proc glm data=project;
 model MarketHealthIndex = SellForGain ZHVI MoM YoY ForecastYoYPctChange NegativeEquity Delinquency DaysOnMarket Zri QoQ BEPropCount SampleRate MedBE MedPR TotalAmountofNE TotalNumberofHomesinNE NEDeliquencyRate;
run;
ods graphics off;

*multiple regression analysis (after the variable selection);
proc reg data=project Rsquare;
 model MarketHealthIndex = SellForGain MoM YoY ForecastYoYPctChange NegativeEquity Delinquency DaysOnMarket Zri / stb corrb;
run;


*LDA;
proc import datafile = "C:/Users/a/Desktop/BIA-652/Project/after_normalization_binary.csv" 
 out=work.classification dbms=csv replace;
 run;
*set the format of the classification;
proc format;         
 value grade 0="Not Healthy" 1="Healthy";
run;
data classification;
 set classification;
 format MarketHealthIndex grade.;
run;
*LDA function;
proc discrim data=classification distance pool=yes outcross=class1 crossvalidate listerr posterr;  
priors equal;
class MarketHealthIndex;
var SellForGain MoM YoY ForecastYoYPctChange NegativeEquity Delinquency DaysOnMarket Zri;
ods output LinearDiscFunc=LinearDiscFunc;     *output the parameter of the discriminant function;
run;
