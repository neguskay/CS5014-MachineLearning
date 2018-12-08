'''
UNIVERSITY OF ST ANDREWS
CS5014 - MACHINE LEARNING

PRACTICAL P1 - PREDICTING ENERGY USE OF APPLIANCES

STUDENT ID: 170027939

@ Main.py
-Extracts data from CSV as pandas data frame
-Plots a Correlation matrix as a heat map
-Cleans and prunes data i.e. removes relatively non-impactful data
-Separates data into Training, Validation and testing data-sets
-Trains Linear Regression Algorithm and tests it accordingly
-Prints out regression stats such as MSE, R-MSE, R-Squared and MAE

'''

import pip as pp
import numpy as np
import csv
import pandas as pd
#import pandas.rpy.common as com
import collections
import matplotlib
import mglearn
import plotly
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.utils import check_array
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from matplotlib.dates import strpdate2num, num2date

'''
from pandas.plotting import scatter_matrix
# Set up the dictionaries for the data columns
# Load Data into "energy_data"

columns = defaultdict(list)


with open('energydata_complete.csv') as csvfile:
    energy_data = csv.reader(csvfile, delimiter=",")

    headers = next(energy_data)
    column_num = range(len(headers))
    # column_num  = range(0, 15750, 1)
    for column in energy_data:
        #print(column[1])
        for i in column_num:
            columns[headers[i]].append(column[i])
columns = dict(columns)

'''


# Pandas dataframe
df_energy = pd.read_csv('/Users/negus/PycharmProjects/CS5014/P1/Main_Energy.csv')

# Init Average lists for House Temp and Humidity
Average_HouseTemp = []
Average_HouseHumidity = []
# print(energy_data)


# Average house temperature and Humidity
for i in range(len(df_energy['Date'])):
    avgTemp = ((df_energy['TempKitchen'][i] + df_energy['TempParentRoom'][i] +
                df_energy['TempLivingRoom'][i] + df_energy['TempLaundry'][i] +
                df_energy['TempOffice'][i] + df_energy['TempBathroom'][i] +
                df_energy['TempIronRoom'][i] + df_energy['TempTeenageRoom'][i])/8)
    Average_HouseTemp.append(avgTemp)

    avgHumidity = ((df_energy['HumKitchen'][i] + df_energy['HumLivingRoom'][i] +
                    df_energy['HumLaundry'][i] + df_energy['HumOffice'][i] +
                    df_energy['HumBathroom'][i] + df_energy['HumIronRoom'][i] +
                    df_energy['HumTeenageRoom'][i] + df_energy['HumParentRoom'][i])/8)
    Average_HouseHumidity.append(avgHumidity)

AvgHouseTemp = pd.DataFrame(Average_HouseTemp, columns=['TempHouseAvg'])
AvgHouseHum = pd.DataFrame(Average_HouseHumidity, columns=['HumHouseAvg'])
df_energy = pd.concat([df_energy, AvgHouseTemp, AvgHouseHum], axis=1)

print(type(df_energy))
#print(df_energy)


# Set Training/Testing Data Ranges
TrainingData_start_Index = 0
TrainingData_end_Index = 16000
Validation_start_Index = 14000


# Extract data into columns
# Generate and slice data into 3 sets
# That is: Training set, Validation set and Testing set
# Training Data
Consumption_Date_Train = df_energy['Date'][:TrainingData_end_Index]  # Time stamp
Consumption_Appliances_Train = df_energy['AppliancesConsumption'][:TrainingData_end_Index]
Consumption_Lights_Train = df_energy['LightsConsumption'][:TrainingData_end_Index]
Temp_Kitchen_Train = df_energy['TempKitchen'][:TrainingData_end_Index]
Hum_Kitchen_Train = df_energy['HumKitchen'][:TrainingData_end_Index]
Temp_LivingRoom_Train = df_energy['TempLivingRoom'][:TrainingData_end_Index]
Hum_LivingRoom_Train = df_energy['HumLivingRoom'][:TrainingData_end_Index]
Temp_Laundry_Train = df_energy['TempLaundry'][:TrainingData_end_Index]
Hum_Laundry_Train = df_energy['HumLaundry'][:TrainingData_end_Index]
Temp_Office_Train = df_energy['TempOffice'][:TrainingData_end_Index]
Hum_Office_Train = df_energy['HumOffice'][:TrainingData_end_Index]
Temp_Bathroom_Train = df_energy['TempBathroom'][:TrainingData_end_Index]
Hum_Bathroom_Train = df_energy['HumBathroom'][:TrainingData_end_Index]
Temp_OutsideNorth_Train = df_energy['TempOusideNorth'][:TrainingData_end_Index]
Hum_OutsideNorth_Train = df_energy['HumOutsideNorth'][:TrainingData_end_Index]
Temp_IronRoom_Train = df_energy['TempIronRoom'][:TrainingData_end_Index]
Hum_IronRoom_Train = df_energy['HumIronRoom'][:TrainingData_end_Index]
Temp_TeenageRoom_Train = df_energy['TempTeenageRoom'][:TrainingData_end_Index]
Hum_TeenageRoom_Train = df_energy['HumTeenageRoom'][:TrainingData_end_Index]
Temp_ParentRoom_Train = df_energy['TempParentRoom'][:TrainingData_end_Index]
Hum_ParentRoom_Train = df_energy['HumParentRoom'][:TrainingData_end_Index]
Temp_FromStation_Train = df_energy['TempFromStation'][:TrainingData_end_Index]
Pressure_FromStation_Train = df_energy['PressureFromStation'][:TrainingData_end_Index]
Hum_FromStation_Train = df_energy['HumidityFromStation'][:TrainingData_end_Index]
Windspeed_FromStation_Train = df_energy['WindspeedFromStation'][:TrainingData_end_Index]
Visibility_FromStation_Train = df_energy['VisibilityFromStation'][:TrainingData_end_Index]
TDewpoint_FromStation_Train = df_energy['Tdewpoint'][:TrainingData_end_Index]
RandomVar1_Train = df_energy['RV1'][:TrainingData_end_Index]
RandomVar2_Train = df_energy['RV2'][:TrainingData_end_Index]
TempAvgHouse_Train = df_energy['TempHouseAvg'][:TrainingData_end_Index]
HumAvgHouse_Train = df_energy['HumHouseAvg'][:TrainingData_end_Index]

# Test print
print(type(Consumption_Lights_Train))
print(Consumption_Lights_Train)

# Validation Data
Consumption_Date_Val = df_energy['Date'][Validation_start_Index:TrainingData_end_Index]  # Time stamp
Consumption_Appliances_Val = df_energy['AppliancesConsumption'][Validation_start_Index:TrainingData_end_Index]
Consumption_Lights_Val = df_energy['LightsConsumption'][Validation_start_Index:TrainingData_end_Index]
Temp_Kitchen_Val = df_energy['TempKitchen'][Validation_start_Index:TrainingData_end_Index]
Hum_Kitchen_Val = df_energy['HumKitchen'][Validation_start_Index:TrainingData_end_Index]
Temp_LivingRoom_Val = df_energy['TempLivingRoom'][Validation_start_Index:TrainingData_end_Index]
Hum_LivingRoom_Val = df_energy['HumLivingRoom'][Validation_start_Index:TrainingData_end_Index]
Temp_Laundry_Val = df_energy['TempLaundry'][Validation_start_Index:TrainingData_end_Index]
Hum_Laundry_Val = df_energy['HumLaundry'][Validation_start_Index:TrainingData_end_Index]
Temp_Office_Val = df_energy['TempOffice'][Validation_start_Index:TrainingData_end_Index]
Hum_Office_Val = df_energy['HumOffice'][Validation_start_Index:TrainingData_end_Index]
Temp_Bathroom_Val = df_energy['TempBathroom'][Validation_start_Index:TrainingData_end_Index]
Hum_Bathroom_Val = df_energy['HumBathroom'][Validation_start_Index:TrainingData_end_Index]
Temp_OutsideNorth_Val = df_energy['TempOusideNorth'][Validation_start_Index:TrainingData_end_Index]
Hum_OutsideNorth_Val = df_energy['HumOutsideNorth'][Validation_start_Index:TrainingData_end_Index]
Temp_IronRoom_Val = df_energy['TempIronRoom'][Validation_start_Index:TrainingData_end_Index]
Hum_IronRoom_Val = df_energy['HumIronRoom'][Validation_start_Index:TrainingData_end_Index]
Temp_TeenageRoom_Val = df_energy['TempTeenageRoom'][Validation_start_Index:TrainingData_end_Index]
Hum_TeenageRoom_Val = df_energy['HumTeenageRoom'][Validation_start_Index:TrainingData_end_Index]
Temp_ParentRoom_Val = df_energy['TempParentRoom'][Validation_start_Index:TrainingData_end_Index]
Hum_ParentRoom_Val = df_energy['HumParentRoom'][Validation_start_Index:TrainingData_end_Index]
Temp_FromStation_Val = df_energy['TempFromStation'][Validation_start_Index:TrainingData_end_Index]
Pressure_FromStation_Val = df_energy['PressureFromStation'][Validation_start_Index:TrainingData_end_Index]
Hum_FromStation_Val = df_energy['HumidityFromStation'][Validation_start_Index:TrainingData_end_Index]
Windspeed_FromStation_Val = df_energy['WindspeedFromStation'][Validation_start_Index:TrainingData_end_Index]
Visibility_FromStation_Val = df_energy['VisibilityFromStation'][Validation_start_Index:TrainingData_end_Index]
TDewpoint_FromStation_Val = df_energy['Tdewpoint'][Validation_start_Index:TrainingData_end_Index]
RandomVar1_Val = df_energy['RV1'][Validation_start_Index:TrainingData_end_Index]
RandomVar2_Val = df_energy['RV2'][Validation_start_Index:TrainingData_end_Index]
TempAvgHouse_Val = df_energy['TempHouseAvg'][Validation_start_Index:TrainingData_end_Index]
HumAvgHouse_Val = df_energy['HumHouseAvg'][Validation_start_Index:TrainingData_end_Index]

# Test Data
Consumption_Date_Test = df_energy['Date'][Validation_start_Index:]  # Time stamp
Consumption_Appliances_Test = df_energy['AppliancesConsumption'][Validation_start_Index:]
Consumption_Lights_Test = df_energy['LightsConsumption'][Validation_start_Index:]
Temp_Kitchen_Test = df_energy['TempKitchen'][Validation_start_Index:]
Hum_Kitchen_Test = df_energy['HumKitchen'][Validation_start_Index:]
Temp_LivingRoom_Test = df_energy['TempLivingRoom'][Validation_start_Index:]
Hum_LivingRoom_Test = df_energy['HumLivingRoom'][Validation_start_Index:]
Temp_Laundry_Test = df_energy['TempLaundry'][Validation_start_Index:]
Hum_Laundry_Test = df_energy['HumLaundry'][Validation_start_Index:]
Temp_Office_Test = df_energy['TempOffice'][Validation_start_Index:]
Hum_Office_Test = df_energy['HumOffice'][Validation_start_Index:]
Temp_Bathroom_Test = df_energy['TempBathroom'][Validation_start_Index:]
Hum_Bathroom_Test = df_energy['HumBathroom'][Validation_start_Index:]
Temp_OutsideNorth_Test = df_energy['TempOusideNorth'][Validation_start_Index:]
Hum_OutsideNorth_Test = df_energy['HumOutsideNorth'][Validation_start_Index:]
Temp_IronRoom_Test = df_energy['TempIronRoom'][Validation_start_Index:]
Hum_IronRoom_Test = df_energy['HumIronRoom'][Validation_start_Index:]
Temp_TeenageRoom_Test = df_energy['TempTeenageRoom'][Validation_start_Index:]
Hum_TeenageRoom_Test = df_energy['HumTeenageRoom'][Validation_start_Index:]
Temp_ParentRoom_Test = df_energy['TempParentRoom'][Validation_start_Index:]
Hum_ParentRoom_Test = df_energy['HumParentRoom'][Validation_start_Index:]
Temp_FromStation_Test = df_energy['TempFromStation'][Validation_start_Index:]
Pressure_FromStation_Test = df_energy['PressureFromStation'][Validation_start_Index:]
Hum_FromStation_Test = df_energy['HumidityFromStation'][Validation_start_Index:]
Windspeed_FromStation_Test = df_energy['WindspeedFromStation'][Validation_start_Index:]
Visibility_FromStation_Test = df_energy['VisibilityFromStation'][Validation_start_Index:]
TDewpoint_FromStation_Test = df_energy['Tdewpoint'][Validation_start_Index:]
RandomVar1_Test = df_energy['RV1'][Validation_start_Index:]
RandomVar2_Test = df_energy['RV2'][Validation_start_Index:]
TempAvgHouse_Test = df_energy['TempHouseAvg'][Validation_start_Index:]
HumAvgHouse_Test = df_energy['HumHouseAvg'][Validation_start_Index:]


# Used to plot several charts for initial analysis
def plot_Analysischarts(x_list, y_list, x_label, y_label, title):
    # print('no')
    plt.scatter(x_list, y_list, label=title, color='k', s=1, marker='x')
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    z = np.polyfit(x_list, y_list, 1)
    p = np.poly1d(z)
    plt.plot(x_list, p(x_list), "r--")
    # plt.legend()
    plt.show()


'''
# Compute Correlation Matrix 
correlation = df_energy.corr()
# grr = scatter_matrix(df_energy, c=df_energy, figsize=(15, 15), marker='o',
# hist_kwds={'bins': 20}, s=60, alpha=.8,cmap=mglearn.cm3)
sb.heatmap(correlation, xticklabels='auto', yticklabels='auto')
plt.show()
'''

# Plot Pair-Plots For Initial Analysis
#sns_df= sb.load_dataset(df_energy)
sb.pairplot(df_energy[["AppliancesConsumption", "LightsConsumption", "TempHouseAvg", "HumHouseAvg"]])  # Too Powerful
#sb.pairplot(df_energy)  # Too Powerful
plt.show()
# print(np.max(y))
print('done')

# Plot Charts by calling plot_AnalysisCharts method
plot_Analysischarts(Consumption_Appliances_Train, TempAvgHouse_Train, 'Appliance Consumption /KWh', 'Average House Temperature /Celcius', 'App Consumption V  Avg Temperature')
plot_Analysischarts(Consumption_Lights_Train, TempAvgHouse_Train, 'Lights Consumption /KWh', 'Average House Temperature /Celcius', 'Lights Consumption V  Avg Temperature')
plot_Analysischarts(Consumption_Appliances_Train, HumAvgHouse_Train, 'Appliance Consumption /KWh', 'Average House Humidity /%', 'App Consumption V  Avg Humidity Percentage')
plot_Analysischarts(Consumption_Lights_Train, HumAvgHouse_Train, 'Lights Consumption /KWh', 'Average House Humidity /%', 'Lights Consumption V  Avg Humidity Percentage')


# Preparing the Inputs and Subsets for Learning
# Subset 1 = 'All' good data (except for ones removed )
# Subset 2 = 'All' data with averages to replace room data(i.e. room temp and humidity)
# Subset 3 = 'Indoor' factors only with Light Consumption
# Subset 4 = 'Outdoor' factors only without Light Consumption

# Subset 1
df_Subset1_Train = pd.concat([Consumption_Appliances_Train ,Consumption_Lights_Train ,Temp_Kitchen_Train ,
                              Hum_Kitchen_Train , Temp_LivingRoom_Train , Hum_LivingRoom_Train, Temp_Laundry_Train ,Hum_Laundry_Train,
                              Temp_Office_Train ,Hum_Office_Train, Temp_Bathroom_Train ,Hum_Bathroom_Train ,Temp_OutsideNorth_Train ,
                              Hum_OutsideNorth_Train ,Temp_IronRoom_Train ,Hum_IronRoom_Train ,Temp_TeenageRoom_Train ,Hum_TeenageRoom_Train,
                              Temp_ParentRoom_Train ,Hum_ParentRoom_Train,Temp_FromStation_Train ,Pressure_FromStation_Train,
                            Hum_FromStation_Train ,Windspeed_FromStation_Train, Visibility_FromStation_Train ,TDewpoint_FromStation_Train],
                             axis=1)

df_Subset1_Val = pd.concat([Consumption_Appliances_Val ,Consumption_Lights_Val,Temp_Kitchen_Val ,
                              Hum_Kitchen_Val , Temp_LivingRoom_Val , Hum_LivingRoom_Val, Temp_Laundry_Val,Hum_Laundry_Val,
                              Temp_Office_Val,Hum_Office_Val, Temp_Bathroom_Val,Hum_Bathroom_Val,Temp_OutsideNorth_Val,
                              Hum_OutsideNorth_Val,Temp_IronRoom_Val,Hum_IronRoom_Val,Temp_TeenageRoom_Val,Hum_TeenageRoom_Val,
                              Temp_ParentRoom_Val,Hum_ParentRoom_Val,Temp_FromStation_Val,Pressure_FromStation_Val,
                            Hum_FromStation_Val,Windspeed_FromStation_Val, Visibility_FromStation_Val,TDewpoint_FromStation_Val],
                             axis=1)

df_Subset1_Test = pd.concat([Consumption_Appliances_Test ,Consumption_Lights_Test,Temp_Kitchen_Test ,
                              Hum_Kitchen_Test, Temp_LivingRoom_Test, Hum_LivingRoom_Test, Temp_Laundry_Test,Hum_Laundry_Test,
                              Temp_Office_Test ,Hum_Office_Test, Temp_Bathroom_Test,Hum_Bathroom_Test,Temp_OutsideNorth_Test ,
                              Hum_OutsideNorth_Test ,Temp_IronRoom_Test ,Hum_IronRoom_Test,Temp_TeenageRoom_Test ,Hum_TeenageRoom_Test,
                              Temp_ParentRoom_Test,Hum_ParentRoom_Test,Temp_FromStation_Test ,Pressure_FromStation_Test,
                            Hum_FromStation_Test ,Windspeed_FromStation_Test, Visibility_FromStation_Test ,TDewpoint_FromStation_Test],
                             axis=1)

# Subset 2
df_Subset2_Train = pd.concat([ Consumption_Lights_Train,Consumption_Appliances_Train,
                                 Temp_OutsideNorth_Train, TempAvgHouse_Train, HumAvgHouse_Train,
                                 TDewpoint_FromStation_Train, Visibility_FromStation_Train,
                                 Pressure_FromStation_Train, Hum_FromStation_Train, Windspeed_FromStation_Train,
                                 Hum_OutsideNorth_Train, Temp_FromStation_Train], axis=1)
                                # Consumption_Appliances_Train, Consumption_Date_Train,

df_Subset2_Val = pd.concat([ Consumption_Lights_Val,Consumption_Appliances_Val,
                                 Temp_OutsideNorth_Val, TempAvgHouse_Val, HumAvgHouse_Val,
                                 TDewpoint_FromStation_Val, Visibility_FromStation_Val,
                                 Pressure_FromStation_Val, Hum_FromStation_Val, Windspeed_FromStation_Val,
                                 Hum_OutsideNorth_Val, Temp_FromStation_Val], axis=1)
                                # Consumption_Date_Val, Consumption_Appliances_Val,

df_Subset2_Test = pd.concat([Consumption_Lights_Test, Consumption_Appliances_Test,
                                 Temp_OutsideNorth_Test, TempAvgHouse_Test, HumAvgHouse_Test,
                                 TDewpoint_FromStation_Test, Visibility_FromStation_Test,
                                 Pressure_FromStation_Test, Hum_FromStation_Test, Windspeed_FromStation_Test,
                                 Hum_OutsideNorth_Test, Temp_FromStation_Test], axis=1)
                                # Consumption_Date_Test, Consumption_Appliances_Test,

# Subset 3
df_Subset3_Train = pd.concat([Consumption_Appliances_Train ,Consumption_Lights_Train ,Temp_Kitchen_Train ,
                              Hum_Kitchen_Train , Temp_LivingRoom_Train , Hum_LivingRoom_Train, Temp_Laundry_Train ,Hum_Laundry_Train,
                              Temp_Office_Train ,Hum_Office_Train, Temp_Bathroom_Train ,Hum_Bathroom_Train,
                              Temp_IronRoom_Train ,Hum_IronRoom_Train ,Temp_TeenageRoom_Train ,Hum_TeenageRoom_Train,
                              Temp_ParentRoom_Train ,Hum_ParentRoom_Train], axis=1)

df_Subset3_Val = pd.concat([Consumption_Appliances_Val ,Consumption_Lights_Val,Temp_Kitchen_Val ,
                              Hum_Kitchen_Val , Temp_LivingRoom_Val , Hum_LivingRoom_Val, Temp_Laundry_Val,Hum_Laundry_Val,
                              Temp_Office_Val,Hum_Office_Val, Temp_Bathroom_Val,Hum_Bathroom_Val
                              ,Temp_IronRoom_Val,Hum_IronRoom_Val,Temp_TeenageRoom_Val,Hum_TeenageRoom_Val,
                              Temp_ParentRoom_Val,Hum_ParentRoom_Val],axis=1)

df_Subset3_Test = pd.concat([Consumption_Appliances_Test ,Consumption_Lights_Test,Temp_Kitchen_Test ,
                              Hum_Kitchen_Test, Temp_LivingRoom_Test, Hum_LivingRoom_Test, Temp_Laundry_Test,Hum_Laundry_Test,
                              Temp_Office_Test ,Hum_Office_Test, Temp_Bathroom_Test,Hum_Bathroom_Test,
                              Temp_IronRoom_Test ,Hum_IronRoom_Test,Temp_TeenageRoom_Test ,Hum_TeenageRoom_Test,
                              Temp_ParentRoom_Test,Hum_ParentRoom_Test], axis=1)

# Subset 4
df_Subset4_Train = pd.concat([Consumption_Appliances_Train, Temp_FromStation_Train ,Pressure_FromStation_Train,Temp_OutsideNorth_Test ,
                              Hum_OutsideNorth_Test ,
                            Hum_FromStation_Train ,Windspeed_FromStation_Train, Visibility_FromStation_Train ,TDewpoint_FromStation_Train],
                             axis=1)

df_Subset4_Val = pd.concat([Consumption_Appliances_Val ,Temp_FromStation_Val,Pressure_FromStation_Val,Temp_OutsideNorth_Test ,
                              Hum_OutsideNorth_Test,
                            Hum_FromStation_Val,Windspeed_FromStation_Val, Visibility_FromStation_Val,TDewpoint_FromStation_Val],
                             axis=1)

df_Subset4_Test = pd.concat([Consumption_Appliances_Test ,Temp_OutsideNorth_Test ,
                              Hum_OutsideNorth_Test ,Temp_FromStation_Test ,Pressure_FromStation_Test,
                            Hum_FromStation_Test ,Windspeed_FromStation_Test, Visibility_FromStation_Test ,TDewpoint_FromStation_Test],
                             axis=1)

'''                                
print('Training Data')
print(df_energy_Train_AVG)
print(type(df_energy_Train_AVG))
print('Validation Data')
print(df_energy_Val_AVG)
print(type(df_energy_Val_AVG))
print('Test Data')
print(df_energy_Test_AVG)
print(type(df_energy_Test_AVG))
print()
'''
#sb.pairplot(df_energy_Train_AVG)
#plt.show()


# Training First Model
# Linear Regression Model

# Linear Regression
# root mean squared error, r2-score, mean absolute error, mean absolute percentage error

def mean_absolute_percentage_error(y_true, y_pred):
    ##y_pred.reshape(-1, 1)
    #y_true.reshape(-1, 1)
    #y_true, y_pred = check_array(y_true, y_pred)
    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true):
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def analyse_SimpleLinearRegression(x_list_Train, y_List_train, x_list_test, y_list_test, Subset_Name):
    print('TRAINING WITH: '+ Subset_Name)
    linreg = LinearRegression()
    linreg.fit(x_list_Train, y_List_train)
    y_hat = linreg.predict(x_list_test)
    print('MSE = ', mean_squared_error(y_list_test, y_hat))
    print(linreg.coef_)
    print('Mean Squared Error = ', mean_squared_error(y_list_test, y_hat))
    print('R-Squared(Variance) Score = ', r2_score(y_list_test, y_hat))
    print('Mean Absolute Error = ', mean_absolute_error(y_list_test, y_hat))
    print('Mean Absolute Error % = ', mean_absolute_percentage_error(y_list_test, y_hat))


analyse_SimpleLinearRegression(df_Subset1_Train, Consumption_Appliances_Train,df_Subset1_Test, Consumption_Appliances_Test, 'SUBSET 1')
print()
analyse_SimpleLinearRegression(df_Subset2_Train, Consumption_Appliances_Train,df_Subset2_Test, Consumption_Appliances_Test, 'SUBSET 2')
print()
analyse_SimpleLinearRegression(df_Subset3_Train, Consumption_Appliances_Train,df_Subset3_Test, Consumption_Appliances_Test, 'SUBSET 3')
print()
analyse_SimpleLinearRegression(df_Subset4_Train, Consumption_Appliances_Train,df_Subset4_Test, Consumption_Appliances_Test, 'SUBSET 4')
print()

'''
TO TRY: EXTENSIONS
- SVM Radial Kernel
- Gradient Boosting
- Random Forest Modelling


'''

