import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

data = pd.read_csv("investments_VC/investments_VC.csv",encoding="ANSI")
data.dropna(axis=0,how='all',inplace=True)
data.dropna(axis=1,how='all',inplace=True)

data.name.fillna('null',inplace=True)
data.homepage_url.fillna('null',inplace=True)
data['founded_year']=data.founded_year.fillna(0).astype(int)

data.columns=[i.replace(' ','') for i in data.columns]
data.funding_total_usd=data.funding_total_usd.map(lambda x:x.replace(',','').replace('-','0')).astype('float')

#%%
plt.hist(data[data.founded_year>0].founded_year,80)
plt.yscale('log')
plt.title('# of startups each year')
plt.xlabel('year')
plt.ylabel('# of startups (log scale)')
plt.grid()
plt.savefig('#startups.png')

#%%
startYear=2000
endYear = 2009
thresh = 60

df={}
for year,group in data.groupby('founded_year'):
    if startYear<=year<=endYear:
        df[str(year)]=group.groupby('market').market.count()

df=pd.DataFrame(df)
df.sort_values(str(endYear),ascending=False,inplace=True)
a=df[np.sum(df>thresh,axis=1)>0]
print(len(a))

sns.heatmap(a,linewidths=.1)
plt.title("areas of statrtups "+str(startYear)+"-"+str(endYear))
plt.xlabel('year')
plt.ylabel('areas of startups')
plt.savefig("areas-"+str(endYear+1)+".png",bbox_inches='tight',dpi=200)

#%%
startYear=1980
endYear = 1999
thresh = 23

df={}
for i in range(startYear,endYear+1):
    df[str(i)]={}
filtered = data[(data.founded_year>=startYear) & (data.founded_year<=endYear)]
for d,y in zip(filtered['category_list'],filtered['founded_year']):
    y=str(y)
    if type(d)!=str:
        continue
    for i in d.split('|'):
        if i in df[y]:
            df[y][i]+=1
        elif i!='':
            df[y][i]=1

df=pd.DataFrame(df)
df.fillna(0,inplace=True)
df.sort_values(str(endYear),ascending=False,inplace=True)
# df=df.T

a=df[np.sum(df>thresh,axis=1)>0]

print(len(a))
sns.heatmap(a,linewidths=.1)
plt.title("categories of statrtups "+str(startYear)+"-"+str(endYear))
plt.xlabel('year')
plt.ylabel('category of startups')
plt.savefig("cat-"+str(endYear+1)+".png",bbox_inches='tight',dpi=200)

#%%
startYear=1980
endYear = 2013

df={}
for i in range(startYear,endYear+1):
    df[str(i)]={}
filtered = data[(data.founded_year>=startYear) & (data.founded_year<=endYear)]
for d,y in zip(filtered['category_list'],filtered['founded_year']):
    y=str(y)
    if type(d)!=str:
        continue
    for i in d.split('|'):
        if i in df[y]:
            df[y][i]+=1
        elif i!='':
            df[y][i]=1

df=pd.DataFrame(df)
df.fillna(0,inplace=True)
df.sort_values(str(endYear),ascending=False,inplace=True)
df.columns=df.columns.astype(int)

plt.plot(df.T['Software'],label='Software')
plt.plot(df.T['Mobile'],label='Mobile')
# plt.plot(df.T['Games'],label='Games')
plt.plot(df.T['Social Media'],label='Social Media')
plt.plot(df.T['Biotechnology'],label='Biotechnology')
plt.plot(df.T['Semiconductors'],label='Semiconductors')
plt.yscale('log')
plt.xticks(range(1980,2020,10))
plt.xlabel('year')
plt.ylabel('# of startups')
plt.title('# of some categories of startups over time')
plt.grid()
plt.legend()

plt.savefig('over-time.png')

#%%
f=data[(1980<data['founded_year'])]
f=f.groupby('founded_year').agg('sum').funding_total_usd
plt.plot(f)
plt.yscale('log')
plt.title('invest on startups each year')
plt.xlabel('year')
plt.ylabel('$ invest on startups (log scale)')
plt.grid()
plt.savefig('invest-startups.png',bbox_inches='tight',dpi=200)

#%%

data.groupby(['market','founded_year']).agg('sum').funding_total_usd
startYear=2010
endYear = 2013
thresh = 0.8e9

df={}
for year,group in data.groupby('founded_year'):
    if startYear<=year<=endYear:
        df[str(year)]=group.groupby('market').funding_total_usd.sum()

df=pd.DataFrame(df).fillna(0)
df.sort_values(str(endYear),ascending=False,inplace=True)
a=df[np.sum(df>thresh,axis=1)>0]
print(len(a))

a=np.log(a+1)
sns.heatmap(a,linewidths=.1)
plt.title("invest areas of statrtups "+str(startYear)+"-"+str(endYear))
plt.xlabel('year')
plt.ylabel('areas of startups')
plt.savefig("invest-areas-"+str(endYear+1)+".png",bbox_inches='tight',dpi=200)

#%%

startYear=2010
endYear = 2013
thresh = 1.3e9

df={}
for i in range(startYear,endYear+1):
    df[str(i)]={}
filtered = data[(data.founded_year>=startYear) & (data.founded_year<=endYear)]
for d,y,im in zip(filtered['category_list'],filtered['founded_year'],filtered["funding_total_usd"]):
    y=str(y)
    if type(d)!=str:
        continue
    for i in d.split('|'):
        if i in df[y]:
            df[y][i]+=im
        elif i!='':
            df[y][i]=im

df=pd.DataFrame(df).fillna(0)
df.sort_values(str(endYear),ascending=False,inplace=True)
a=df[np.sum(df>thresh,axis=1)>0]
a=np.log(a+1)

print(len(a))

sns.heatmap(a,linewidths=.1)
plt.title("categories of statrtups "+str(startYear)+"-"+str(endYear))
plt.xlabel('year')
plt.ylabel('category of startups')
plt.savefig("invest-cat-"+str(endYear+1)+".png",bbox_inches='tight',dpi=200)

#%%
startYear=1980
endYear = 2013

df={}
for i in range(startYear,endYear+1):
    df[str(i)]={}
filtered = data[(data.founded_year>=startYear) & (data.founded_year<=endYear)]
for d,y,im in zip(filtered['category_list'],filtered['founded_year'],filtered["funding_total_usd"]):
    y=str(y)
    if type(d)!=str:
        continue
    for i in d.split('|'):
        if i in df[y]:
            df[y][i]+=im
        elif i!='':
            df[y][i]=im

df=pd.DataFrame(df)
df.fillna(0,inplace=True)
df.sort_values(str(endYear),ascending=False,inplace=True)
df.columns=df.columns.astype(int)

plt.plot(df.T['Software'],label='Software')
plt.plot(df.T['Mobile'],label='Mobile')
# plt.plot(df.T['Games'],label='Games')
plt.plot(df.T['Social Media'],label='Social Media')
plt.plot(df.T['Biotechnology'],label='Biotechnology')
plt.plot(df.T['Semiconductors'],label='Semiconductors')
plt.yscale('log')
plt.xticks(range(1980,2020,10))
plt.xlabel('year')
plt.ylabel('invest on startups')
plt.title('invest on some categories of startups over time')
plt.grid()
plt.legend()

plt.savefig('invest-over-time.png')

#%%
a=data.groupby("country_code").name.count().sort_values(ascending=False)[:10]
plt.bar(a.index,a)
plt.yscale('log')
plt.title('# of startups in each country')
plt.xlabel('country')
plt.ylabel('# of startups')
plt.savefig(".png",bbox_inches='tight',dpi=200)

#%%
# d=data.dropna(subset=['status'])
d=data.copy()
d.status=d.status.fillna('closed')

a=d[d['status']=='operating'].groupby('market').status.count()
b=d.groupby('market').status.count()
i=d.groupby('market').funding_total_usd.sum()
c=pd.concat([a,b,i],axis=1)
c.columns=['success','total','invest']
c.fillna(0,inplace=True)
c.index=[i.replace(' ','') for i in c.index]
c['success']/=c['total']

plt.scatter(c.total,c.success,color='gray',alpha=0.2,s=5)
plt.scatter(c.loc['Software']['total'],c.loc['Software']['success'],s=15,label='Software')
plt.scatter(c.loc['Mobile']['total'],c.loc['Mobile']['success'],s=15,label='Mobile')
plt.scatter(c.loc['Games']['total'],c.loc['Games']['success'],s=15,label='Games')
plt.scatter(c.loc['SocialMedia']['total'],c.loc['SocialMedia']['success'],s=15,label='Social Media')
plt.scatter(c.loc['Biotechnology']['total'],c.loc['Biotechnology']['success'],s=15,label='Biotechnology')
plt.scatter(c.loc['Semiconductors']['total'],c.loc['Semiconductors']['success'],s=15,label='Semiconductors')
plt.xscale('log')
plt.legend()
plt.xlim((10,5000))
plt.ylim((0.4,1))
plt.xlabel('# of startups')
plt.ylabel('success rate')
plt.title('success rate vs # of startups in each field')
plt.savefig("successVSall.png",bbox_inches='tight',dpi=200)

#%%
import plotly.express as px
import plotly
# %plotly inline

c['field']='other'
indexes=c.sort_values('invest',ascending=False)[:5].index
c.loc[indexes,'field']=indexes


fig=px.scatter(x=c['total'],y=c['success']
               ,size=c['invest']
               ,hover_data=[c.index]
               ,color=c.field
               ,log_x=True)
fig.update_layout(
    title='success rate vs # of startups in each field'
    ,title_x=0.5
    ,xaxis_title='# of startups'
    ,yaxis_title='success rate')
plotly.offline.plot(fig)


