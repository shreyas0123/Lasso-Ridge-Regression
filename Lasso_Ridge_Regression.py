############################## problem1 ############################################
# Multilinear Regression with Regularization using L1 and L2 norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data
startups_data = pd.read_csv("E:/DATA SCIENCE ASSIGNMENT/Class And Assignment Dataset/Asss/Lasso Ridge Regression/50_Startups.csv")

#EDA
#Eliminating state coloumn
startups_data = startups_data.iloc[:,[0,1,2,4]]
# Rearrange the order of the variables
startups_data = startups_data.iloc[:, [3, 0, 1, 2]]
startups_data.columns

#Rename the column
startups_data = startups_data.rename(columns = {'R&D Spend':'RD_Spend','Marketing Spend':'Marketing_Spend'})


# Correlation matrix 
a = startups_data.corr()
a

# EDA
a1 = startups_data.describe()

# Sctter plot and histogram between variables
sns.pairplot(startups_data) 
# Preparing the model on train data 
model_train = smf.ols("Profit ~ RD_Spend + Administration + Marketing_Spend ", data = startups_data).fit()
model_train.summary()

# Prediction
pred = model_train.predict(startups_data)
# Error
resid  = pred - startups_data.Profit
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse

# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###
from sklearn.linear_model import Lasso
help(Lasso)

lasso = Lasso(alpha = 0.13, normalize = True)

lasso.fit(startups_data.iloc[:, 1:], startups_data.Profit)

# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(startups_data.columns[1:]))

lasso.alpha

pred_lasso = lasso.predict(startups_data.iloc[:, 1:])

# Adjusted r-square
lasso.score(startups_data.iloc[:, 1:], startups_data.Profit)

# RMSE
np.sqrt(np.mean((pred_lasso - startups_data.Profit)**2))


### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(startups_data.iloc[:, 1:], startups_data.Profit)

# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(startups_data.columns[1:]))

rm.alpha

pred_rm = rm.predict(startups_data.iloc[:, 1:])

# Adjusted r-square
rm.score(startups_data.iloc[:, 1:], startups_data.Profit)

# RMSE
np.sqrt(np.mean((pred_rm - startups_data.Profit)**2))


### ELASTIC NET REGRESSION ###
from sklearn.linear_model import ElasticNet 
help(ElasticNet)
enet = ElasticNet(alpha = 0.4)

enet.fit(startups_data.iloc[:, 1:], startups_data.Profit) 

# Coefficients values for all the independent vairbales
enet.coef_
enet.intercept_

plt.bar(height = pd.Series(enet.coef_), x = pd.Series(startups_data.columns[1:]))

enet.alpha

pred_enet = enet.predict(startups_data.iloc[:, 1:])

# Adjusted r-square
enet.score(startups_data.iloc[:, 1:], startups_data.Profit)

# RMSE
np.sqrt(np.mean((pred_enet - startups_data.Profit)**2))


####################

# Lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(startups_data.iloc[:, 1:], startups_data.Profit)


lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred = lasso_reg.predict(startups_data.iloc[:, 1:])

# Adjusted r-square#
lasso_reg.score(startups_data.iloc[:, 1:], startups_data.Profit)

# RMSE
np.sqrt(np.mean((lasso_pred - startups_data.Profit)**2))



# Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(startups_data.iloc[:, 1:], startups_data.Profit)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(startups_data.iloc[:, 1:])

# Adjusted r-square#
ridge_reg.score(startups_data.iloc[:, 1:], startups_data.Profit)

# RMSE
np.sqrt(np.mean((ridge_pred - startups_data.Profit)**2))



# ElasticNet Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

enet = ElasticNet()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 5)
enet_reg.fit(startups_data.iloc[:, 1:], startups_data.Profit)

enet_reg.best_params_
enet_reg.best_score_

enet_pred = enet_reg.predict(startups_data.iloc[:, 1:])

# Adjusted r-square
enet_reg.score(startups_data.iloc[:, 1:], startups_data.Profit)

# RMSE
np.sqrt(np.mean((enet_pred - startups_data.Profit)**2))

############################## problem 2 ###############################################
# Multilinear Regression with Regularization using L1 and L2 norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data
Computer_Data = pd.read_csv("E:/DATA SCIENCE ASSIGNMENT/Class And Assignment Dataset/Asss/Lasso Ridge Regression/Computer_Data.csv")

#EDA
#Eliminating 1 coloumn
Computer_Data = Computer_Data.iloc[:,1:]

#creat dummy variables
Computer_Data = pd.get_dummies(Computer_Data)
#colnames
Computer_Data.columns

# Correlation matrix 
a = Computer_Data.corr()
a

# EDA
a1 = Computer_Data.describe()

# Sctter plot and histogram between variables
sns.pairplot(Computer_Data) 
# Preparing the model on train data 
model_train = smf.ols('price ~ speed + hd + ram + screen + ads + trend + cd_no + cd_yes + multi_no + multi_yes + premium_no + premium_yes', data = Computer_Data).fit()
model_train.summary()

# Prediction
pred = model_train.predict(Computer_Data)
# Error
resid  = pred - Computer_Data.price
# RMSE value for data o docum
rmse = np.sqrt(np.mean(resid * resid))
rmse

# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###
from sklearn.linear_model import Lasso
help(Lasso)

lasso = Lasso(alpha = 0.13, normalize = True)

lasso.fit(Computer_Data.iloc[:, 1:], Computer_Data.price)

# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(Computer_Data.columns[1:]))

lasso.alpha

pred_lasso = lasso.predict(Computer_Data.iloc[:, 1:])

# Adjusted r-square
lasso.score(Computer_Data.iloc[:, 1:], Computer_Data.price)

# RMSE
np.sqrt(np.mean((pred_lasso - Computer_Data.price)**2))


### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(Computer_Data.iloc[:, 1:], Computer_Data.price)

# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(Computer_Data.columns[1:]))

rm.alpha

pred_rm = rm.predict(Computer_Data.iloc[:, 1:])

# Adjusted r-square
rm.score(Computer_Data.iloc[:, 1:], Computer_Data.price)

# RMSE
np.sqrt(np.mean((pred_rm - Computer_Data.price)**2))


### ELASTIC NET REGRESSION ###
from sklearn.linear_model import ElasticNet 
help(ElasticNet)
enet = ElasticNet(alpha = 0.4)

enet.fit(Computer_Data.iloc[:, 1:], Computer_Data.price) 

# Coefficients values for all the independent vairbales
enet.coef_
enet.intercept_

plt.bar(height = pd.Series(enet.coef_), x = pd.Series(Computer_Data.columns[1:]))

enet.alpha

pred_enet = enet.predict(Computer_Data.iloc[:, 1:])

# Adjusted r-square
enet.score(Computer_Data.iloc[:, 1:], Computer_Data.price)

# RMSE
np.sqrt(np.mean((pred_enet - Computer_Data.price)**2))


####################

# Lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(Computer_Data.iloc[:, 1:], Computer_Data.price)


lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred = lasso_reg.predict(Computer_Data.iloc[:, 1:])

# Adjusted r-square#
lasso_reg.score(Computer_Data.iloc[:, 1:], Computer_Data.price)

# RMSE
np.sqrt(np.mean((lasso_pred - Computer_Data.price)**2))

# Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(startups_data.iloc[:, 1:], startups_data.Profit)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(startups_data.iloc[:, 1:])

# Adjusted r-square#
ridge_reg.score(startups_data.iloc[:, 1:], startups_data.Profit)

# RMSE
np.sqrt(np.mean((ridge_pred - startups_data.Profit)**2))



# ElasticNet Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

enet = ElasticNet()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 5)
enet_reg.fit(startups_data.iloc[:, 1:], startups_data.Profit)

enet_reg.best_params_
enet_reg.best_score_

enet_pred = enet_reg.predict(startups_data.iloc[:, 1:])

# Adjusted r-square
enet_reg.score(startups_data.iloc[:, 1:], startups_data.Profit)

# RMSE
np.sqrt(np.mean((enet_pred - startups_data.Profit)**2))

####################################### problem3 ###################################
# Multilinear Regression with Regularization using L1 and L2 norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data
car_data = pd.read_csv("E:/DATA SCIENCE ASSIGNMENT/Class And Assignment Dataset/Asss/Lasso Ridge Regression/ToyotaCorolla.csv",encoding = "ISO-8859-1")

#EDA
#Selcting required coloumn
car_data = car_data.iloc[:,[2,3,6,8,12,13,15,16,17]]

#Getting coloumn
car_data.columns

# Correlation matrix 
a = car_data.corr()
a

# EDA
a1 = car_data.describe()

# Sctter plot and histogram between variables
sns.pairplot(car_data) 
# Preparing the model on train data 
model_train = smf.ols("Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight ", data = car_data).fit()
model_train.summary()

# Prediction
pred = model_train.predict(car_data)
# Error
resid  = pred - car_data.Price
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse

# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###
from sklearn.linear_model import Lasso
help(Lasso)

lasso = Lasso(alpha = 0.13, normalize = True)

lasso.fit(car_data.iloc[:, 1:], car_data.Price)

# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(car_data.columns[1:]))

lasso.alpha

pred_lasso = lasso.predict(car_data.iloc[:, 1:])

# Adjusted r-square
lasso.score(car_data.iloc[:, 1:], car_data.Price)

# RMSE
np.sqrt(np.mean((pred_lasso - car_data.Price)**2))


### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(car_data.iloc[:, 1:], car_data.Price)

# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(car_data.columns[1:]))

rm.alpha

pred_rm = rm.predict(car_data.iloc[:, 1:])

# Adjusted r-square
rm.score(car_data.iloc[:, 1:], car_data.Price)

# RMSE
np.sqrt(np.mean((pred_rm - car_data.Price)**2))


### ELASTIC NET REGRESSION ###
from sklearn.linear_model import ElasticNet 
help(ElasticNet)
enet = ElasticNet(alpha = 0.4)

enet.fit(car_data.iloc[:, 1:], car_data.Price) 

# Coefficients values for all the independent vairbales
enet.coef_
enet.intercept_

plt.bar(height = pd.Series(enet.coef_), x = pd.Series(car_data.columns[1:]))

enet.alpha

pred_enet = enet.predict(car_data.iloc[:, 1:])

# Adjusted r-square
enet.score(car_data.iloc[:, 1:], car_data.Price)

# RMSE
np.sqrt(np.mean((pred_enet - car_data.Price)**2))


####################

# Lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(car_data.iloc[:, 1:], car_data.Price)


lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred = lasso_reg.predict(car_data.iloc[:, 1:])

# Adjusted r-square#
lasso_reg.score(car_data.iloc[:, 1:], car_data.Price)

# RMSE
np.sqrt(np.mean((lasso_pred - car_data.Price)**2))



# Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(car_data.iloc[:, 1:], car_data.Price)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(car_data.iloc[:, 1:])

# Adjusted r-square#
ridge_reg.score(car_data.iloc[:, 1:], car_data.Price)

# RMSE
np.sqrt(np.mean((ridge_pred - car_data.Price)**2))



# ElasticNet Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

enet = ElasticNet()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 5)
enet_reg.fit(car_data.iloc[:, 1:], car_data.Price)

enet_reg.best_params_
enet_reg.best_score_

enet_pred = enet_reg.predict(car_data.iloc[:, 1:])

# Adjusted r-square
enet_reg.score(car_data.iloc[:, 1:], car_data.Price)

# RMSE
np.sqrt(np.mean((enet_pred - car_data.Price)**2))

########################## problem4 ###################################
# Multilinear Regression with Regularization using L1 and L2 norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data
Life_expectencey_data = pd.read_csv("E:/DATA SCIENCE ASSIGNMENT/Class And Assignment Dataset/Asss/Lasso Ridge Regression/Life_expectencey_LR.csv")

#EDA
#remove NA values
Life_expectencey_data.isna().sum()
Life_expectencey_data = Life_expectencey_data.dropna()
#Eliminating coloumn
Life_expectencey_data = Life_expectencey_data.drop(["Country","Status"],axis = 1 )
# Rearrange the order of the variables
Life_expectencey_data = Life_expectencey_data.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,0]]
Life_expectencey_data.columns

# Correlation matrix 
a = Life_expectencey_data.corr()
a

# EDA
a1 = Life_expectencey_data.describe()

# Sctter plot and histogram between variables
sns.pairplot(Life_expectencey_data) 
# Preparing the model on train data 
model_train = smf.ols(" Life_expectancy ~ Adult_Mortality + infant_deaths + Alcohol + percentage_expenditure + Hepatitis_B + Measles  + BMI + under_five_deaths + Polio + Total_expenditure + Diphtheria + HIV_AIDS + GDP + Population + thinness + thinness_yr + Income_composition + Schooling + Year", data = Life_expectencey_data).fit()
model_train.summary()

# Prediction
pred = model_train.predict(Life_expectencey_data)
# Error
resid  = pred - Life_expectencey_data.Life_expectancy
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse

# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###
from sklearn.linear_model import Lasso
help(Lasso)

lasso = Lasso(alpha = 0.13, normalize = True)

lasso.fit(Life_expectencey_data.iloc[:, 1:], Life_expectencey_data.Life_expectancy)

# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(Life_expectencey_data.columns[1:]))

lasso.alpha

pred_lasso = lasso.predict(Life_expectencey_data.iloc[:, 1:])

# Adjusted r-square
lasso.score(Life_expectencey_data.iloc[:, 1:], Life_expectencey_data.Life_expectancy)

# RMSE
np.sqrt(np.mean((pred_lasso - Life_expectencey_data.Life_expectancy)**2))


### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(Life_expectencey_data.iloc[:, 1:], Life_expectencey_data.Life_expectancy)

# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(Life_expectencey_data.columns[1:]))

rm.alpha

pred_rm = rm.predict(Life_expectencey_data.iloc[:, 1:])

# Adjusted r-square
rm.score(Life_expectencey_data.iloc[:, 1:], Life_expectencey_data.Life_expectancy)

# RMSE
np.sqrt(np.mean((pred_rm - Life_expectencey_data.Life_expectancy)**2))


### ELASTIC NET REGRESSION ###
from sklearn.linear_model import ElasticNet 
help(ElasticNet)
enet = ElasticNet(alpha = 0.4)

enet.fit(Life_expectencey_data.iloc[:, 1:], Life_expectencey_data.Life_expectancy) 

# Coefficients values for all the independent vairbales
enet.coef_
enet.intercept_

plt.bar(height = pd.Series(enet.coef_), x = pd.Series(Life_expectencey_data.columns[1:]))

enet.alpha

pred_enet = enet.predict(Life_expectencey_data.iloc[:, 1:])

# Adjusted r-square
enet.score(Life_expectencey_data.iloc[:, 1:], Life_expectencey_data.Life_expectancy)

# RMSE
np.sqrt(np.mean((pred_enet - Life_expectencey_data.Life_expectancy)**2))


####################

# Lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(Life_expectencey_data.iloc[:, 1:], Life_expectencey_data.Life_expectancy)


lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred = lasso_reg.predict(Life_expectencey_data.iloc[:, 1:])

# Adjusted r-square#
lasso_reg.score(Life_expectencey_data.iloc[:, 1:], Life_expectencey_data.Life_expectancy)

# RMSE
np.sqrt(np.mean((lasso_pred - Life_expectencey_data.Life_expectancy)**2))



# Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(Life_expectencey_data.iloc[:, 1:], Life_expectencey_data.Life_expectancy)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(Life_expectencey_data.iloc[:, 1:])

# Adjusted r-square#
ridge_reg.score(Life_expectencey_data.iloc[:, 1:], Life_expectencey_data.Life_expectancy)

# RMSE
np.sqrt(np.mean((ridge_pred - Life_expectencey_data.Life_expectancy)**2))



# ElasticNet Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

enet = ElasticNet()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 5)
enet_reg.fit(Life_expectencey_data.iloc[:, 1:], Life_expectencey_data.Life_expectancy)

enet_reg.best_params_
enet_reg.best_score_

enet_pred = enet_reg.predict(Life_expectencey_data.iloc[:, 1:])

# Adjusted r-square
enet_reg.score(Life_expectencey_data.iloc[:, 1:], Life_expectencey_data.Life_expectancy)

# RMSE
np.sqrt(np.mean((enet_pred - Life_expectencey_data.Life_expectancy)**2))

##################################### END #######################################

