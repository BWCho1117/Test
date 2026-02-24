
# coding: utf-8

# In[24]:


from tools.misctools import csv_data_extraction_attendee as csv_tool 
from importlib import reload
import numpy as np
reload(csv_tool)
 
mlq = csv_tool.AttendeesManager(work_folder=r'C:\\Users\\cristian\\Research\\WorkData') 
mlq.set_fields_MLQ2021()
fname = 'C:/Users/dw29/Dropbox (Heriot-Watt University Team)/RES_EPS_Quantum_Photonics_Lab/Events/MLQ2021/20210221_attendees.csv'

df = mlq.import_csv(fname)

df.head()


# Some of the countries are entered in long form

# In[33]:


print(df['Country'].unique()[13], '\n' , df['Country'].unique()[19])
df['Country'][df['Country'].str.contains('Croatia')] = 'Croatia'
df['Country'][df['Country'].str.contains('Iran')] = 'Iran'


# # Field Pie

# In[21]:


df.plot(figsize=(20, 20));
df['Field:'].dropna().astype('str').value_counts().plot.pie(fontsize=22)


# # Occupation Pie

# In[19]:


df.plot(figsize=(20, 20));
df['Occupation:'].dropna().astype('str').value_counts().plot.pie(fontsize=22)


# # Country Pie

# In[20]:


df.plot(figsize=(20, 20))
plot = df['Country'].dropna().astype('str').value_counts().plot(kind='pie', fontsize=11, title='')

