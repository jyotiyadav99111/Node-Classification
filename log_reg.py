import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


# Data Loading
df = pd.read_csv("https://raw.githubusercontent.com/jyotiyadav99111/Node-Classification/master/wiki_edges.csv", header = None)
df_full = pd.read_csv("https://raw.githubusercontent.com/jyotiyadav99111/Node-Classification/master/wiki_edges.csv", header = None)

df.columns = ['A','B']
df_full.columns = ['node', 'label']

df = df[df['A'] != df['B']]
df_array = np.asarray(df)

# Data Preprocessing
data = {}
for line in df_array:
    if line[0] in data:
        data[line[0]].append(line[1])
    else:
        data[line[0]] = [line[1]]

new_data = [''] * df_full.shape[0]
for key in data.keys():
    new_data[key] = data[key]

X=  pd.DataFrame.from_dict(data, orient='index')
df_full['col']=  new_data


# Data Seg
Y = df_full['label']
X = df_full.drop('label', axis = 1)

mlb = MultiLabelBinarizer()
final_X = pd.DataFrame(mlb.fit_transform(X['col']))
final_X['node'] = X['node']


# Split test and triaining data set
X_train, X_test, y_train, y_test = train_test_split(final_X, Y, test_size=0.2, random_state=42)


# Model
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
prediction = logisticRegr.predict(X_test)


# Results
results = {}
results['f1_score'] = f1_score(y_test, prediction, average=None)
results['acc'] = accuracy_score(y_test,prediction)
print(results)
