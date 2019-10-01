#PARAS MEHAN
#ROLL NO.2018062
#PROBABLITY AND STATISTICS
#ASSIGNMENT - 2 
#Script is divided into parts.


#-----------------------------------------------------------------------------------------------------#
#PART-1	Data Extraction from data_journals.csv,found.txt

import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None 


data1=pd.read_csv('data_journals.csv',sep=';',usecols=['Title','SJR','H index','Total Docs. (2017)','Total Docs. (3years)','Total Refs.','Total Cites (3years)','Citable Docs. (3years)','Cites / Doc. (2years)','Ref. / Doc.'])
data2=pd.read_csv('found.txt',sep=';',header=None,names=['Title','H index','Impact Factor'])

#-----------------------------------------------------------------------------------------------------#
#PART-2 Merging Data  

def merge(df):
	global data2
	for i in range(len(data2['Title'])):
		if df['Title']==data2['Title'][i]:
			return data2['Impact Factor'][i]
		
	return -1

data1['Impact Factor']=data1.apply(merge,axis=1)

#-----------------------------------------------------------------------------------------------------#
#PART-3 Data cleaning

l=[]
for i in range(len(data1['Impact Factor'])):
	if (data1['Impact Factor'][i]+1)<10**-6:
		l.append(i)
	#print(data1['Impact Factor'][i])
l.append(1441)
for i in l:
	data1.drop(i,inplace=True)


li=list(data1['SJR'].to_dict().keys())
#print(li)

for i in li:
	for j in range(len(data1['SJR'][i])):
		if data1['SJR'][i][j]==',':
			data1['SJR'][i]=int(data1['SJR'][i][:j]+data1['SJR'][i][j+1:])
			#print(data1['SJR'][i],type(data1['SJR'][i]))
			break
			
		#data1['SJR'][i]=int(data1['SJR'][i])

	for j in range(len(data1['Cites / Doc. (2years)'][i])):
		if data1['Cites / Doc. (2years)'][i][j]==',':
			data1['Cites / Doc. (2years)'][i]=data1['Cites / Doc. (2years)'][i][:j]+'.'+data1['Cites / Doc. (2years)'][i][j+1:]
			data1['Cites / Doc. (2years)'][i]=float(data1['Cites / Doc. (2years)'][i])
			#print(data1['SJR'][i],type(data1['SJR'][i]))
			break
	
	for j in range(len(data1['Ref. / Doc.'][i])):
		if data1['Ref. / Doc.'][i][j]==',':
			data1['Ref. / Doc.'][i]=data1['Ref. / Doc.'][i][:j]+'.'+data1['Ref. / Doc.'][i][j+1:]
			data1['Ref. / Doc.'][i]=float(data1['Ref. / Doc.'][i])
			#print(data1['SJR'][i],type(data1['SJR'][i]))
			break        

test_data=data1.sample(int(len(data1)*0.2))

l2=list(test_data['Citable Docs. (3years)'].to_dict().keys())
for i in l2:
	data1.drop(i,inplace=True)

#-----------------------------------------------------------------------------------------------------#
#PART-4 Processing Data

op=[]
op.append(np.array(data1['SJR'].astype('int').to_numpy()))
op.append(data1['H index'].to_numpy())
op.append(data1['Total Docs. (2017)'].to_numpy())
op.append(data1['Total Docs. (3years)'].to_numpy())
op.append(data1['Total Refs.'].to_numpy())
op.append(data1['Total Cites (3years)'].to_numpy())
op.append(data1['Citable Docs. (3years)'].to_numpy())
op.append(data1['Cites / Doc. (2years)'].astype('float64').to_numpy())
op.append(data1['Ref. / Doc.'].astype('float64').to_numpy())
y=np.array(data1['Impact Factor'])


op2=np.array(test_data['SJR'].astype('int').to_numpy())
op2=np.vstack((op2,test_data['H index'].to_numpy()))
op2=np.vstack((op2,test_data['Total Docs. (2017)'].to_numpy()))
op2=np.vstack((op2,test_data['Total Docs. (3years)'].to_numpy()))
op2=np.vstack((op2,test_data['Total Refs.'].to_numpy()))
op2=np.vstack((op2,test_data['Total Cites (3years)'].to_numpy()))
op2=np.vstack((op2,test_data['Citable Docs. (3years)'].to_numpy()))
op2=np.vstack((op2,test_data['Cites / Doc. (2years)'].astype('float64').to_numpy()))
op2=np.vstack((op2,test_data['Ref. / Doc.'].astype('float64').to_numpy()))
yt=test_data['Impact Factor'].to_numpy()

opl=[]
for i in range(1,2**9):
	x=str(bin(i))
	x=x[2:]
	if x.count('1')>=2:
		while(len(x)<9):
			x='0'+x
		opl.append(x)

#-----------------------------------------------------------------------------------------------------#
#PART-5 Using Multivariate Regression using To predict Linear Regression

def pred(x):
	global y
	xt=x.transpose()
	a=np.matmul(x,xt)
	b=np.linalg.inv(a)
	c=np.matmul(b,x)
	return np.matmul(c,y)

def error(x,p):
	global yt
	x=x.transpose()
	#print(len(x),p.shape)
	yo=np.matmul(x,p)
	yo=yo-yt
	e=0.0
	for i in yo:
		e+=i**2
	return (sum(abs(yo))/len(x),e/len(x))

ans=[]
#p=[opl[0]]
#x3=[]
for i in range(len(opl)):
	
	x2=[];
	x4=[]
	for j in range(9):
		if opl[i][j]=='1':
			x2.append(op[j])
			x4.append(op2[j])
	#print(x2)        
	
	x3=np.array(x2[0])
	x5=np.array(x4[0])
	for j in range(1,len(x2)):
		x3=np.vstack((x3,x2[j]))
		x5=np.vstack((x5,x4[j]))
	#print(x5.shape)
	#print(i,x5)
	a=error(x5,pred(x3))
	#print(type(a))
	a=list(a)
	a.append(opl[i])
	#print(a)
	#break
	ans.append(a)

#-----------------------------------------------------------------------------------------------------#
#PART-6 Finding Minimum and Reporting Answers to "Answers.txt" file

def s1(x):
	return x[0]

def s2(x):
	return x[1]


def fs(x):
	s=""
	fl=['SJR','H index','Total Docs. (2017)','Total Docs. (3years)','Total Refs.','Total Cites (3years)','Citable Docs. (3years)','Cites / Doc. (2years)','Ref. / Doc.']
	for i in range(9):
		if x[i]=='1':
			s=s+fl[i]+','

	return s

file=open('Answers.txt',"a")

file.write("\nProbability and Statictics\nAssignment-2\n\n1)For Minimum Mean Absolute Error\nMean Absolute error    Mean Squared Error     Selected Combination of Features")
ans.sort(key=s1)

#print(ans[:5])
for i in range(5):
	file.write("\n%1.5f"%(ans[i][0])+'                '+"%1.5f"%(ans[i][1])+"                "+fs(ans[i][2]))

ans.sort(key=s2)
file.write("\n\n2)For Minimum Mean Squared Error\nMean Absolute error    Mean Squared Error     Selected Combination of Features")

#print(ans[:5])
for i in range(5):
	file.write("\n%1.5f"%(ans[i][0])+"                "+"%1.5f"%(ans[i][1])+"                "+fs(ans[i][2]))
 		