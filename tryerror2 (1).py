import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pylab
from scipy.stats import norm
import scipy.stats as st

data = pd.read_csv("cell2celltrain.csv")

#data.head()
data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
data=data[ data['AgeHH1']> 0 ]
data=data[ data['AgeHH2']> 0 ]
# dropna function drops rows with any nan entries
#any to delete if any column contains nan
#axis=0 to delete rows and not columns
# true to change in the given data
rows=len(data)
#total number of colmuns
String = data.select_dtypes(exclude=[np.number])
#filtering only string columns
Stirng=String.drop(columns=['ServiceArea'])
#removing unwanted columns
Number = data.select_dtypes(exclude=[np.object])
#filtering only intger columns
Number=Number.drop(columns=['CustomerID'])

#removing unwanted columns
#probabilty of a string in a feild of strings

#for column in String.columns[1:]:
#    print(String[column].value_counts()/x)
#    print('***********')
    
#joint probability between two string columns
jointcols=String[['CreditRating','Churn']]
jointprob=jointcols.groupby('Churn')['CreditRating'].value_counts()/rows
#print(jointprob)

#conditional probability between two string columns
condcols=String[['Homeownership','Churn']]
condprob=condcols.groupby('Churn')['Homeownership'].value_counts()/ condcols.groupby('Churn')['Homeownership'].count()
#print(condprob)

print('*********')

#histogram

#rangecolum =Number['UnansweredCalls']
#range=(rangecolum.max())-(rangecolum.min())
#plt.hist('UnansweredCalls',bins=5)
#histogram=Number.hist(figsize=(25,25))
    
#pdf   
#for column in Number.columns[1:]:
#    H, R =np.histogram(Number[column], bins=20,density=True) 
#    plt.figure()
#    plt.plot(R[1:] ,H)
#    
#    #cdf
#    pinsize = math.ceil(R[2]-R[1])
#    arraysum = (H[:1]+H[1:])*0.5
#    cdf = pinsize *arraysum
#    output =  cdf.cumsum()
#    plt.figure()
#    plt.plot(R[2:] ,output)
#    
print('####################')




#JOINTPDF      
jointpdf=Number[['MonthlyRevenue','MonthlyMinutes']]
joint_pdf=jointpdf.groupby('MonthlyMinutes')['MonthlyRevenue'].value_counts()/rows
rangeMonthRev=Number['MonthlyRevenue']
rangeMonthMin=Number['MonthlyMinutes']

X=Number['MonthlyRevenue']
Y=data['MonthlyMinutes']

#h,y1,y2=np.histogram2d(X,Y,bins=(25,25),density=True)
#x,y=np.meshgrid(y1[1:],y2[1:])
#plt.contourf(x,y,h)
#plt.show()

#print(Number.mean())
#print('################################')
#print(Number.var())
#fig.tight_layout()
#plt.hist2d(H,y1,y2)
# (H[:1]+H[1:])*0.5   
#H,R =np.histogram()
#s=pd.Number.Series(['MonthsInService'])
#print(s.plot.kde())
#h2,xr,yr=np.histogram2d
#data_anamolies=data[ data['AgeHH1']> 0 ]
#data_anamolies=data[ data['AgeHH2']> 0 ]
data_anamolies11=data.drop(columns=['CustomerID' ,'ServiceArea'])
#rows_anamo=len(data_anamolies11)
#col_anamo=len(data_anamolies11.columns)
#
corrdata=data_anamolies11.corr()
#corrdata1=data_anamolies11.corr()


i=0
j=0

while (i < len(corrdata) ):
    while( j < len(corrdata) ):
        value=abs(corrdata.iloc[i,j])
        if((value>0) & (value<0.3) ) :
            corrdata.iloc[i,j]='Independent'
        elif( (value>= 0.3) & (value < 0.6)) :
            corrdata.iloc[i,j]='Dependent'
        elif((value>=0.6) & (value<1)) :
            corrdata.iloc[i,j]='Strongly Dependent'
        else :
            corrdata.iloc[i,j]='-_-'
        j=j+1
    i=i+1
    j=0
##########################################################################
hlepful = data.copy() 
checkdata=hlepful[:15000]
checkdata=checkdata.drop(columns=['CustomerID' ,'ServiceArea'])
rest_data=hlepful[15001 : 16000]
churn_true=rest_data.copy() 
churn_true=churn_true.loc[:, churn_true.columns.intersection(['Churn'])]
rest_data1=rest_data.drop(columns=['CustomerID' ,'ServiceArea','Churn'])
checknumber=checkdata.select_dtypes(exclude=[np.object])
checkstring=checkdata.select_dtypes(exclude=[np.number])


                   #fit_pdf #point 3
distributions = [ st.beta,st.expon,st.norm ]
col_name1 =Number[:1]
i=-1
pars_vector1=[]
for column in checknumber.columns[1:]:
    mles1 =[]
    i=i+1
    for distribution in distributions:
      U1, I1 =np.histogram(checknumber[column], bins=20,density=True)
      pars1 = distribution.fit(checknumber[column])
      # Separate parts of parameters
      arg = pars1[:-2]
      loc = pars1[-2]
      scale = pars1[-1]
      # Calculate fitted PDF and error with fit in distribution
      H2 = distribution.pdf(I1[1:], loc=loc, scale=scale, *arg)
      error=H2- U1
      meanerros3=pow(error,2).mean()
      mles1.append(meanerros3)
    disindex=mles1.index(min(mles1))
    col_name1.iloc[0,i]=disindex
    pars2=distributions[disindex].fit(checknumber[column])
    pars_vector1.append(pars2)

#########################################################################
                     #fit_conditional pdf # point 4
rowyes=checkdata.loc[checkdata.Churn=='Yes']     
rowno=checkdata.loc[checkdata.Churn=='No']    
rowyesnumber = rowyes.select_dtypes(exclude=[np.object])
helpme = rowyesnumber.copy() 
mles=[]
size=len(rowyesnumber)
distrib=[]
col_name=helpme[:1]
col_name.loc[:,:]=0
pars_vector=[]
i=-1
for column in rowyesnumber.columns[1:]:
    
    mles =[]
    i=i+1
    for distribution in distributions:
      U, I =np.histogram(rowyesnumber[column], bins=20,density=True)
      pars = distribution.fit(rowyesnumber[column])
      # Separate parts of parameters
      arg = pars[:-2]
      loc = pars[-2]
      scale = pars[-1]
      # Calculate fitted PDF and error with fit in distribution
      H1 = distribution.pdf(I[1:], loc=loc, scale=scale, *arg)
      error=H1- U
      meanerros2=pow(error,2).mean()
      mles.append(meanerros2)
    disindex=mles.index(min(mles))
    col_name.iloc[0,i]=disindex
    pars1 = distributions[disindex].fit(rowyesnumber[column])
    pars_vector.append(pars1)
################################################################################
    #NOT CHURN 
rowNo=checkdata.loc[checkdata.Churn=='No']         
rowNonumber = rowNo.select_dtypes(exclude=[np.object])
helpmeno =rowNonumber.copy()
mlesNo=[]
size=len(rowNonumber)
distrib=[]
col_nameNo=helpmeno[:1]
col_nameNo.loc[:,:]=0
pars_vector_No=[]
i=-1
for column in rowNonumber.columns[1:]:
    
    mlesNo =[]
    i=i+1
    for distribution in distributions:
      UNo, INo =np.histogram(rowNonumber[column], bins=20,density=True)
      parsNo = distribution.fit(rowNonumber[column])
      # Separate parts of parameters
      argNo = parsNo[:-2]
      locNo = parsNo[-2]
      scaleNo = parsNo[-1]
      # Calculate fitted PDF and error with fit in distribution
      H1No = distribution.pdf(INo[1:], loc=locNo, scale=scaleNo, *argNo)
      errorNo=H1No- UNo
      meanerros2No=pow(error,2).mean()
      mlesNo.append(meanerros2No)
    disindexNo=mlesNo.index(min(mlesNo))
    col_nameNo.iloc[0,i]=disindexNo
    pars1No = distributions[disindexNo].fit(rowNonumber[column])
    pars_vector_No.append(pars1No)
#########################################################################
    #POINT 4 #STRINGS  
    #CHURN YES 
rowyesString = rowyes.select_dtypes(exclude=[np.number])
tabel_cond = []
#for row in range(len(rowyesString) ):
for column in rowyesString.columns[1:] :
    ss =rowyesString[column].value_counts()/len(rowyesString)
    tabel_cond.append(ss)
 # ChuRN  = No    
rowNoString = rowNo.select_dtypes(exclude=[np.number])
tabel_condNo = []
#for row in range(len(rowNoString) ):
for column in rowNoString.columns[1:] :
    ss1 =rowNoString[column].value_counts()/len(rowNoString)
    tabel_condNo.append(ss1)
        
##############################################################################
    #POINT 5
row_RESTnumber1 = rest_data1.select_dtypes(exclude=[np.object]) 
row_RESTString = rest_data1.select_dtypes(exclude=[np.number]) 
# FOR NUMBERS  
i=0
finalrobability = 1
prob = []
prob_Churn=checkdata['Churn'].value_counts()/len(checkdata)
prob_yes=prob_Churn.iloc[1]
for row in range(len(row_RESTnumber1)): 
        i=0
        finalrobability =1
        while(i < len( row_RESTnumber1.columns)-1) :
            typedes = distributions [int(col_name.iloc[0,i])]
            pars=pars_vector[i]  
            arg = pars[:-2]
            loc = pars[-2]
            scale = pars[-1]
            H1 = typedes.pdf(row_RESTnumber1.iloc[row,i], loc=loc, scale=scale, *arg)
            if(H1!=0):
                finalrobability = finalrobability * (H1)
            i=i+1  
        churnH1_Yes = finalrobability *prob_yes 
        prob.append(churnH1_Yes)
        
#######################################################################
#        NOT CHURN
i=0
prob_No =prob_Churn.iloc[0] 
probNo =[]
for row in range(len(row_RESTnumber1)): 
        i=0
        finalrobability_No =1
        while(i < len( row_RESTnumber1.columns)-1) :
            typedes_No = distributions [int(col_nameNo.iloc[0,i])]
            pars_No=pars_vector_No[i]  
            arg_No = pars_No[:-2]
            loc_No = pars_No[-2]
            scale_No = pars_No[-1]
            H1_No = typedes_No.pdf(row_RESTnumber1.iloc[row,i], loc=loc_No, scale=scale_No, *arg_No)
            if(H1_No != 0):    
                finalrobability_No = finalrobability_No * H1_No
            i=i+1

        churnH1_No = finalrobability_No *prob_No 
        probNo.append(churnH1_No)    
#######################################################################################
# for String
probStringyes = 1        
i = 0
FinalprobYes = [] 
FinalprobNo = [] 
for row in range(len(row_RESTString)): 
    i=0
    probStringyes=1
    while(i < len( row_RESTString.columns)-6) :
        value = row_RESTString.iloc[row,i]
        probStringyes=probStringyes* tabel_cond[i][value] 
        i+=1
    ffprob =probStringyes*prob[row]
    FinalprobYes.append(ffprob)
#NOT CHURN 
probStringno = 1        
i = 0
for row in range(len(row_RESTString)): 
    i=0
    probStringno=1
    while(i < len( row_RESTString.columns)-6) :
        value = row_RESTString.iloc[row,i]
        probStringno=probStringno*(tabel_condNo[i][value])
        i+=1        
    ffprobno =probStringyes*probNo[row]
    FinalprobNo.append(ffprobno)        
#################################################################
        # POINT 6
check_churn = [] 
for rows in range(len(prob) ):
    if(FinalprobYes[rows]>=FinalprobNo[rows]):
       check_churn.append('Yes') 
    if(FinalprobYes[rows]<FinalprobNo[rows]):
        check_churn.append('No') 
    
###############################

i=0
c=[]
while(i <len(churn_true)):
    c.append(churn_true.iloc[i,0])
    i+=1
   


count = 0 
for rows in range(len(prob) ):    
    if(check_churn[rows] == c[rows]):
        count+=1 
        
accuracy = count/len(prob)
print(accuracy)
##############################################################################################




















