# -*- coding: utf-8 -*-


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import libraries 
import fastai 
from fastai import * 
from fastai.text import * 
import pandas as pd 
import numpy as np 
from functools import partial 
import io 
import os
from fastai.callbacks.mem import PeakMemMetric

#torch distributed
defaults.device = torch.device('cpu')
from fastai.distributed import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
torch.set_device(args.local_rank)
torch.distributed.init_process_group(backend='gloo', init_method='env://')

from sklearn.datasets import fetch_20newsgroups
dataset = fetch_20newsgroups()
documents = dataset.data

df = pd.DataFrame({'label':dataset.target, 'text':dataset.data})
df = df.reset_index(drop = True)

#df['label'].value_counts()

df['text'] = df['text'].str.replace("[^a-zA-Z]", " ")

import nltk
nltk.download('stopwords') 
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

# tokenization 
tokenized_doc = df['text'].apply(lambda x: x.split())

# remove stop-words 
tokenized_doc = tokenized_doc.apply(lambda x:[item for item in x if 
                                    item not in stop_words])

# de-tokenization 
detokenized_doc = []

for i in range(len(df)):
    t =' '.join(tokenized_doc[i]) 
    detokenized_doc.append(t)

df['text'] = detokenized_doc

from sklearn.model_selection import train_test_split

# split data into training and validation set 
df_trn, df_val = train_test_split(df, stratify = df['label'],  test_size = 0.2, random_state = 12)

df_trn.shape, df_val.shape

# Language model data 
data_lm = TextLMDataBunch.from_df(train_df = df_trn, valid_df = df_val, path = "")

# Classifier model data 
data_clas = TextClasDataBunch.from_df(path = "", train_df = df_trn, 
                                      valid_df = df_val,  
                                      vocab=data_lm.train_ds.vocab, 
                                      bs=32)

learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.2,callback_fns=PeakMemMetric) #.to_distributed(args.local_rank)

print(torch.cuda.device_count())

learn.lr_find()
learn.recorder.plot()

# train the learner object with learning rate = 1e-2 
learn.fit_one_cycle(3, 1e-2)

learn.unfreeze()
learn.fit_one_cycle(3, slice(2e-3/100, 2e-3))

learn.save_encoder('ft_enc')

classifier = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.7) 
classifier.load_encoder('ft_enc')

classifier.fit_one_cycle(1, 1e-2)

classifier.unfreeze()
classifier.fit_one_cycle(3, slice(2e-3/100, 2e-3))

# get predictions 
preds, targets = classifier.get_preds(DatasetType.Valid) 
predictions = np.argmax(preds, axis = 1)

for i in range(10):
  print("The targets and predictions are: ", targets[i], predictions[i])

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2)
#predictions = model.predict(X_test, batch_size=1000)

#LABELS = ['graphics','hockey'] 
LABELS = df['label'].unique()

confusion_matrix = metrics.confusion_matrix(targets, predictions)

plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d", annot_kws={"size": 20});
plt.title("Confusion matrix", fontsize=20)
plt.ylabel('True label', fontsize=20)
plt.xlabel('Predicted label', fontsize=20)
plt.show()

confusion_matrix = pd.crosstab(predictions, targets)

print(confusion_matrix)

#Recall Values:

for i in range(20):
  print("The recall for Class ", i , " is", confusion_matrix[i][i]/(sum(confusion_matrix[i])))