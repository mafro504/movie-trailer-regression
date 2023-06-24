###IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
import random
import itertools
import os


###READ IN DATA

#updated sustain distrib to binary
Trailer8 = pd.read_csv('/Trailer_Dataset9.csv')



###remove dollar signs
cols = Trailer4.columns
Trailer[cols] = Trailer[cols].replace({'\$':''}, regex = True)

print (Trailer['spend_dayof'])

###CHECK DATA
#first few lines look correct
print (Trailer.head())

###data types look correct (all integer or float)
print (Trailer5.info())

###initial stats of data
stat_sum = Trailer5.describe()
stat_sum.to_excel('Stat_Summ_2.xlsx')

###########################################################

###SPEND DISTRIBUTION###

#####calc avg total $ amount dedicated to each phase
dayof_avg = Trailer["spend_dayof"].mean() # $36,966.52
tease_avg = Trailer["spemd_tease"].mean() # $1,896.38
sustain_avg = Trailer["spend_sustain"].mean() # $51,977.39
total_avg = Trailer["spend_total"].mean() # $90,840.29
print("Day of Average:", dayof_avg, "Tease Average:", tease_avg, "Sustain Average:", sustain_avg, "Total Average:", total_avg)

######calc avg proportion of total spend dedicated to each phase
dayof_prop = dayof_avg/total_avg # 0.40693971074540847 40.70%
tease_prop = tease_avg/total_avg # 0.02087593363614173 2.09%
sustain_prop = sustain_avg/total_avg # 0.5721843556184499 57.22%
print("Day of Proportion:", dayof_prop, "Tease Proportion:", tease_prop, "Sustain Proportion:", sustain_prop)

###########################################################

###CORRELATION ANALYSIS###
TrailerCorrs = Trailer['spend_total'].corr(Trailer['spend_dayof'])
TrailerCorrs

CorrMatrix = Trailer2.corr()
CorrMatrix.to_excel('TrailerCorrelations4.xlsx')


# Calculate pearson coefficient and p-value
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

###########################################################

#####REGRESSION ANALYSIS#####


#set seed so analysis is replicable to get consistent results
random.seed(456456)

print (X.info())

##train test split##
#separate data into training and testing sets. set 30% of data to test and 70% to training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#dont necessarily need to run on train and test with small sample (n=98)

##creating and training model##
#create linear regression object
lm = LinearRegression()

#fit/train model on my training set
model = lm.fit(X_train,y_train)

#stepwise regression
reg = sm.OLS(y,X).fit()
summ = reg.summary()
print(summ)

#to account for categorical variables
ROUND 1: 
"""model = ols('paid_trailer_views_scaled ~  spend_total_scaled + spend_sustain_bin + spend_dayof_distrib + allowlisted_Talent + Branded_Reminder + C(Distribution_Type, Treatment(reference="Both")) + C(launch_handle_clean, Treatment(reference="brand")) + C(IMDB_Grouping_Broad, Treatment(reference="Bottom")) + C(TKO_Campaign, Treatment(reference="None")) + C(TKO_TrailerDay, Treatment(reference="None")) + C(Campaign_Length, Treatment(reference="1day")) + C(Trailer_LeadInTime_Weeks, Treatment(reference="1_4weeks"))', data=Trailer8)"""

ROUND 2. vars removed: Branded_Reminder
"""model = ols('paid_trailer_views_scaled ~  spend_total_scaled + spend_sustain_bin + spend_dayof_distrib + allowlisted_Talent + C(Distribution_Type, Treatment(reference="Both")) + C(launch_handle_clean, Treatment(reference="brand")) + C(IMDB_Grouping_Broad, Treatment(reference="Bottom")) + C(TKO_Campaign, Treatment(reference="None")) + C(TKO_TrailerDay, Treatment(reference="None")) + C(Campaign_Length, Treatment(reference="1day")) + C(Trailer_LeadInTime_Weeks, Treatment(reference="1_4weeks"))', data=Trailer8)"""


ROUND 3. vars removed: + C(IMDB_Grouping_Broad, Treatment(reference="Bottom"))
"""model = ols('paid_trailer_views_scaled ~  spend_total_scaled + spend_sustain_bin + spend_dayof_distrib + allowlisted_Talent + C(Distribution_Type, Treatment(reference="Both")) + C(launch_handle_clean, Treatment(reference="brand")) + C(TKO_Campaign, Treatment(reference="None")) + C(TKO_TrailerDay, Treatment(reference="None")) + C(Campaign_Length, Treatment(reference="1day")) + C(Trailer_LeadInTime_Weeks, Treatment(reference="1_4weeks"))', data=Trailer8)"""


ROUND 4. vars removed: + C(Campaign_Length, Treatment(reference="1day"))
"""model = ols('paid_trailer_views_scaled ~  spend_total_scaled + spend_sustain_bin + spend_dayof_distrib + allowlisted_Talent + C(Distribution_Type, Treatment(reference="Both")) + C(launch_handle_clean, Treatment(reference="brand")) + C(TKO_Campaign, Treatment(reference="None")) + C(TKO_TrailerDay, Treatment(reference="None")) + C(Trailer_LeadInTime_Weeks, Treatment(reference="1_4weeks"))', data=Trailer8)"""


ROUND 5. vars removed: + C(Distribution_Type, Treatment(reference="Both"))
"""model = ols('paid_trailer_views_scaled ~  spend_total_scaled + spend_sustain_bin + spend_dayof_distrib + allowlisted_Talent + C(launch_handle_clean, Treatment(reference="brand")) + C(TKO_Campaign, Treatment(reference="None")) + C(TKO_TrailerDay, Treatment(reference="None")) + C(Trailer_LeadInTime_Weeks, Treatment(reference="1_4weeks"))', data=Trailer8)"""


ROUND 6. vars removed:  + C(TKO_Campaign, Treatment(reference="None"))
#model = ols('paid_trailer_views_scaled ~  spend_total_scaled + spend_sustain_bin + spend_dayof_distrib + allowlisted_Talent + C(launch_handle_clean, Treatment(reference="brand")) + C(TKO_TrailerDay, Treatment(reference="None")) + C(Trailer_LeadInTime_Weeks, Treatment(reference="1_4weeks"))', data=Trailer8)
model = ols('log(paid_trailer_views_scaled) ~  spend_total_scaled + spend_sustain_bin + spend_dayof_distrib + allowlisted_Talent + C(launch_handle_clean, Treatment(reference="brand")) + C(TKO_TrailerDay, Treatment(reference="None")) + C(Trailer_LeadInTime_Weeks, Treatment(reference="1_4weeks"))', data=Trailer8)


##WINNER##
#ROUND 14. vars removed: playing around with trailerleadintime vars


fitted_model = model.fit()
step_summ = fitted_model.summary()
print(step_summ)

#round 1: 
#adj Rsq = .486

#round 2: 
#adj Rsq = .493

#round 3: 
#adj Rsq = .485

#round 4: 
#adj Rsq = .481

#round 5: 
#adj Rsq = .471

#round 6: WINNER
#adj Rsq = .482



#running on entire sample
model = lm.fit(X,y)


##model evaluation##

r_sq = model.score(X_train,y_train)
print('coefficient of determination:', r_sq) #0.8858064921137239

#print intercept
print('intercept:', model.intercept_) #-3635198.587818306

#create coefficient DF
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
print('coefficients:', coeff_df)
coeff_df.to_excel('TrailerCoefficients2.xlsx')

#coefficeint for each feature

#pulled standard deviation for each variable in order to calculate standardized coefficients (so that when interpreting results, I am on the same unit level)
standard_dev = np.std(Trailer8['spend_sustain_bin'])
print('standard deviations:', standard_dev)
standard_dev.to_excel('StDevs8.xlsx')

#find p values to get SS
#p-val below .05 indicates groupsa re out of balance
X2 = sm.add_constant(X_train)
est = sm.OLS(y_train, X2)
est2 = est.fit()
TrailerSum = est2.summary()
print('p-values:',est2.summary())

TrailerSum.to_excel('pvals.xlsx')


#ANOVA of categorical variables
"""ANOVA_TKOCampaign = stats.f_oneway(Trailer5['paid_trailer_views_scaled'][Trailer5['TKO_Campaign'] == 'None'],
               Trailer5['paid_trailer_views_scaled'][Trailer5['TKO_Campaign'] == 'Trend Takeover+'],
               Trailer5['paid_trailer_views_scaled'][Trailer5['TKO_Campaign'] == 'Trend Takeover+ & First View'],
               Trailer5['paid_trailer_views_scaled'][Trailer5['TKO_Campaign'] == 'First View'])"""
#print(ANOVA_TKOCampaign)

"""ANOVA_Handle = stats.f_oneway(Trailer5['paid_trailer_views_scaled'][Trailer5['launch_handle_clean'] == 'brand'],
               Trailer5['paid_trailer_views_scaled'][Trailer5['launch_handle_clean'] == 'studio'],
               Trailer5['paid_trailer_views_scaled'][Trailer5['launch_handle_clean'] == 'Movie'])
print(ANOVA_Handle)"""



##transforming coefficients to get to ##x views so its more digestile

#create a dataset with every combination of values of the predictors. one row for each combo of predictors.
#expand.grid creates dataset

df = {
    'launch_handle_clean_Movie': [1, 0],
    'launch_handle_clean_studio': [1, 0],
    #'launch_handle_clean_brand': ['brand'],
    #'TKO_TrailerDay_None': ['None'],
    'TKO_TrailerDay_FV': [1, 0],
    'TKO_TrailerDay_TrendTKO': [1, 0],
    #'Trailer_LeadInTime_Weeks_1_4weeks': ['1_4weeks'],
    'Trailer_LeadInTime_Weeks_5_12weeks': [1, 0],
    'Trailer_LeadInTime_Weeks_13_24weeks': [1, 0],
    'Trailer_LeadInTime_Weeks_25_36weeks': [1, 0],
    'Trailer_LeadInTime_Weeks_52plusweeks': [1, 0],
    'spend_sustain_bin': [1, 0],
    'allowlisted_Talent': [1, 0],
    'spend_total_scaled': [1,0.1,0.3,0.39,3.99,2.2,0.5,0.4,2,3.5,0.2,0.13,2.79,1.47,0.19,0.05,0.21,1.24,2.74,2.48,1.58,0.48,0.16,0.26,1.2,1.25,1.5,1.01,0.07,0.35,0.75,0.49,0.99,0.14,0.25,0.03,0.62,1.49,3.4,2.62,0.34,0.23,0.24,0.45,0.32,0.42,1.38,0.29,0.53,0.09,0.22,0.11,0.06,0.15,0.18,0.41,0.6,0.67,0.76,2.18,0.54],
    'spend_dayof_distrib': [0.99, 0.68, 0.25, 0.08, 0.2, 0.07, 0.21, 0.17, 0.14, 1, 0.75, 0.18, 0.11, 0.03, 0.97, 0.98, 0.24, 0.01, 0.13,
0.32,
0.26,
0.31,
0.33,
0.42,
0.93,
0.4,
0.59,
0.57,
0.34,
0.49,
0.9,
0.67,
0.29,
0.89,
0.7,
0.45,
0.72,
0.15,
0.85,
0.38,
0.91,
0.76],
'paid_trailer_views_scaled': [4.35,1.19,0.26,0.16,3.83,1.39,1.09,0.35,1.97,0.78,2.28,1.43,0.2,1.16,0.17,0.42,0.25,1,1.27,0.09,0.63,1.69,2.43,1.06,0.47,0.5,1.55,1.52,2.2,1.37,1.51,2.78,0.08,2.79,0.3,1.05,0.56,2.03,1.15,0.49,0.21,2.07,3.37,8.65,2.38,0.44,0.91,2.18,1.4,0.51,1.08,0.38,0.12,0.72,1.63,1.32,2.31,1.34,0.9,2.32,0.99,1.66,0.57,2.21,0.14,0.41,1.79,1.41,2.62,1.38,2.17,2.55,2.52,4.03,6.28,5.43,9.88,9.37,0.69,0.59,0.27]
}

print('df type is:', type((float(df['launch_handle_clean_Movie'][0]))))

order = df.keys()
tempdata = pd.DataFrame(itertools.product(*[df[k] for k in order]), columns=order)
#print(tempdata)
print('tempdata:',type(float(tempdata['launch_handle_clean_Movie'][0])))

####then apply the model to predict the outcome for each of these combinations (each row)
#calculating predictions
tempdata['pred'] = model.predict(fitted_model.params, tempdata)
Trailer8['pred'] = model.predict(fitted_model.params, Trailer8)


#print(tempdata)
#print(tempdata['pred'])

tempdata_pred = fitted_model.predict(pd.DataFrame(tempdata))
tempdata_pred = fitted_model.predict(tempdata)
tempdata_pred = model.predict(tempdata)
tempdata_pred = model.predict(tempdata.to_numpy())
tempdata_pred = model.fit(tempdata)

##look for way to exclude new levels/predictors


####then look at average for different value of the main predictor I'm interested in
####group it by variable im interested in and calc mean of predictions. 
####this gives you average prediction for each value of the grouping variable (i.e this will give me avg prediction for TKO and avg predic for non-TKO)
tmp = tempdata.groupby(['spend_total_scaled'])
#tmp = Trailer8.groupby(['Trailer_LeadInTime_Weeks_52plusweeks'])

#print(type(tmp)) #checking to make sure its a DF before using DF syntax to find mean on next line
tmp2 = tmp['pred'].mean()
print(tmp2)

#repeat for each var

####then you compare those and then you can express it as "x" by 
####calculating (TKOprediction - nonTKOprediction)/nonTKOprediction

