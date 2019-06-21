
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib as plt
import pandas as pd


# In[2]:



import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
client_d1e6f87f465c411a96d9c4cf6bb4c760 = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='u_beHA_WDYfc2Mrfvb88URmnfg-l3syWLRQqH60v_maI',
    ibm_auth_endpoint="https://iam.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.eu-geo.objectstorage.service.networklayer.com')

body = client_d1e6f87f465c411a96d9c4cf6bb4c760.get_object(Bucket='deeplearningprojects-donotdelete-pr-ehykmit03xm3cj',Key='whitewine.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

dataset = pd.read_csv(body)
dataset.head()



# In[3]:


dataset


# In[4]:


dataset.corr()


# In[5]:


x=dataset.iloc[:,1:12].values


# In[6]:


x


# In[7]:


y=dataset.iloc[:,12].values


# In[8]:


y


# In[9]:


from keras.utils import to_categorical
y=to_categorical(y)


# In[10]:


y[22]


# In[11]:


from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[12]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[13]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[14]:


def create_model():
    model=Sequential()
    model.add(Dense(input_dim=11,init="random_uniform",activation='relu',output_dim=70))
    model.add(Dense(output_dim=60,init="random_uniform",activation='relu'))#hidden layer
    model.add(Dense(output_dim=50,init="random_uniform",activation='relu'))#hidden layer
    model.add(Dense(output_dim=40,init="random_uniform",activation='relu'))
    model.add(Dense(output_dim=30,init="random_uniform",activation='relu'))
    model.add(Dense(output_dim=10,init='random_uniform',activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model


# In[15]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import pipeline
pipe = pipeline.Pipeline([
    ('rescale', StandardScaler()),
    ('nn', KerasClassifier(build_fn=create_model, epochs=512, batch_size=32))
])


# In[16]:


pipe.fit(x_train, y_train)


# In[17]:


y.reshape(-1, 1)


# In[18]:


y_p=pipe.predict(np.array([[6.5,0.18,0.34,1.6,0.04,43,148,0.9912,3.32,0.59,11.5]]))


# In[19]:


n=max(y_p)


# In[20]:


n


# In[21]:


pipe.named_steps['nn'].model.save('hack.h5')


# In[22]:


get_ipython().system(u'tar -zcvf hack.tgz hack.h5')


# In[23]:


from watson_machine_learning_client import WatsonMachineLearningAPIClient


# In[24]:


wml_credentials={
    "access_key": "FmedMmiwebKPNU7_IvTcS3NtuRs-RwdYXPV14Cfqizk1",
  #"iam_apikey_description": "Auto-generated for key 36198445-9a55-4c9f-9d94-15dac72fed35",
  #"iam_apikey_name": "Service credentials-1",
  #"iam_role_crn": "crn:v1:bluemix:public:iam::::serviceRole:Writer",
  #"iam_serviceid_crn": "crn:v1:bluemix:public:iam-identity::a/d7852154a2ce45c585044d475afaaf92::serviceid:ServiceId-0b1bed91-431b-480c-a6f4-127d66a555ee",
 "instance_id": "b94c7666-8d06-47af-9133-e53330ed55e6",
  "password": "f1228f3b-1aef-4286-a60e-c39a4cea6670",
  "url": "https://eu-gb.ml.cloud.ibm.com",
  "username": "1570de1e-199e-416d-8f3c-086fcecc09a6"
}


# In[25]:


client=WatsonMachineLearningAPIClient(wml_credentials)


# In[26]:


metadata={
    client.repository.ModelMetaNames.NAME:"keras model",
    client.repository.ModelMetaNames.FRAMEWORK_LIBRARIES:[{'name':'keras','version':'2.1.3'}],
    client.repository.ModelMetaNames.FRAMEWORK_NAME:"tensorflow",
    client.repository.ModelMetaNames.FRAMEWORK_VERSION:"1.5"
}


# In[27]:


model_details=client.repository.store_model(model="hack.tgz",meta_props=metadata)


# In[28]:


client.repository.list_models()


# In[29]:


model_uid=model_details['metadata']['guid']#It is retreiving the GUID from the particular model
#In order to deploy our model we have to know our GUID 
model_deploy=client.deployments.create(artifact_uid=model_uid,name="wine")


# In[30]:


scoring_endpoint=client.deployments.get_scoring_url(model_deploy)


# In[31]:


scoring_endpoint

