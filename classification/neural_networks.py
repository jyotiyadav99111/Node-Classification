import pandas as pd
import tensorflow as tf

# Data Loading
df = pd.read_csv("https://raw.githubusercontent.com/jyotiyadav99111/Node-Classification/master/data/wiki/wiki_edges.csv", header = None)
df_full = pd.read_csv("https://raw.githubusercontent.com/jyotiyadav99111/Node-Classification/master/data/wiki/wiki_labels.csv", header = None)

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

dataset = tf.data.Dataset.from_tensor_slices((final_X, Y))

for feat, targ in dataset.take(5):
  print ('Features: {}, Target: {}'.format(feat, targ))
  
train_dataset = dataset.shuffle(len(final_X)).batch(1)


model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation='relu'),
  tf.keras.layers.Dense(50, activation='relu'),
  tf.keras.layers.Dense(17)
])

model.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_dataset, epochs=100)

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)
