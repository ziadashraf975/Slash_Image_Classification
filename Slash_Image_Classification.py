#!/usr/bin/env python
# coding: utf-8

# In[78]:


get_ipython().system('pip install tensorflow==2.2.0')


# In[79]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, classification_report
from tensorflow.keras import models, layers, optimizers
from tensorflow.python.keras.saving import hdf5_format
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
import h5py, itertools, collections
import itertools
get_ipython().run_line_magic('matplotlib', 'inline')

##################
# Verifications:
#################
print('GPU is used.' if len(tf.config.list_physical_devices('GPU')) > 0 else 'GPU is NOT used.')
print("Tensorflow version: " + tf.__version__)


# In[80]:


image_size = (64, 64)
# We define the batch size
batch_size = 64

# Create an image generator with a fraction of images reserved for validation:
image_generator = ImageDataGenerator(zoom_range=[0.5,1.0],
                                     brightness_range=[0.2,1.0],
                                     rotation_range=45,
                                     horizontal_flip=True
                                     )

# Now, we create a training data iterator by creating batchs of images of the same size as 
# defined previously, i.e., each image is resized in a 64x64 pixels format.
train_ds =  DirectoryIterator(
    "slashdata",
    image_generator,
    class_mode='categorical',
    seed=1337,
    target_size=image_size,
    batch_size=batch_size,
    subset = 'training',
)
val_gen = ImageDataGenerator(validation_split=0.2)
val_ds = DirectoryIterator(
    "slashdata",
    val_gen,
    class_mode='categorical',
    seed=1337,
    target_size=image_size,
    batch_size=batch_size,
    subset = 'validation',
    shuffle=False
)

# We save the list of classes (labels).
class_names = list(train_ds.class_indices.keys())

# We also save the number of labels.
num_classes = train_ds.num_classes


# In[81]:


print(class_names)
print(num_classes)


# In[82]:


counter=collections.Counter(train_ds.labels)
v = [ [class_names[item[0]],item[1]]  for item in counter.items()]
df = pd.DataFrame(data=v, columns=['index','value'])
g = sns.catplot(x='index', y= 'value',  data=df, kind='bar', 
                legend=False,height=4,aspect=4,saturation=1)
(g.despine(top=False,right=False))
plt.xlabel("Classes")
plt.ylabel("#images")
plt.title("Distribution of images per class")
plt.xticks(rotation='vertical')
plt.show()

#####################################
######### Show sample of images.
#####################################
plt.figure(figsize=(20, 16))
images = []
labels = []
for itr in train_ds.next():
    for i in range(30):
        if len(images) < 30:
            images.append(itr[i].astype("uint8"))
        else:
            labels.append(list(itr[i]).index(1))

for i in range(len(images)):
    ax = plt.subplot(5, 6, i + 1)
    plt.imshow(images[i])
    plt.title(class_names[labels[i]].replace('_',' ') +' ('+str(int(labels[i]))+')')
    plt.axis("off")


# In[83]:


model = models.Sequential()
model.add(keras.Input(shape=image_size + (3,))) 
model.add(layers.experimental.preprocessing.Rescaling(1./255))
model.add(layers.Conv2D(32, (3,3), padding='SAME', activation='relu'))
model.add(layers.Conv2D(32, (3,3), padding='SAME', activation='relu'))
model.add(layers.MaxPool2D())
model.add(layers.Dropout(0.2))
model.add(layers.BatchNormalization())
model.add(layers.GaussianNoise(0.1))
model.add(layers.Conv2D(64, (3,3), padding='SAME', activation='relu'))
model.add(layers.Conv2D(64, (3,3), padding='SAME', activation='relu'))
model.add(layers.MaxPool2D())
#model.add(layers.Dropout(0.2))
model.add(layers.SpatialDropout2D(0.2))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3,3), padding='SAME', activation='relu'))
#Dense part
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu', activity_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(256, activation='relu', activity_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(layers.Dense(128, activation='relu', activity_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(7, activation='softmax', activity_regularizer=tf.keras.regularizers.l2(0.001)))
# Print a summary of the model
model.summary()


# In[84]:


model.compile(optimizer=optimizers.SGD(lr=0.01, momentum=0.7),
              loss='CategoricalCrossentropy',
              metrics=['accuracy'])


# In[85]:


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6)
lr_on_plateu = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.01, patience=4, min_lr=0.001)
mcp_save = tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')


# In[86]:


history = model.fit(
    train_ds, epochs = 200, batch_size=64, steps_per_epoch=train_ds.samples/batch_size,
                    validation_data=val_ds, validation_steps=val_ds.samples/batch_size, callbacks=[early_stopping, lr_on_plateu, mcp_save]
)


# In[87]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.figure(figsize=(15,4))

ax1 = plt.subplot(1, 2, 1)
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.fill_between(epochs, loss,val_loss,color='g',alpha=.1)

plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

ax2 = plt.subplot(1, 2, 2)
plt.plot(epochs, acc, label='Training accuracy')
plt.plot(epochs, val_acc, label='Validation accuracy')
plt.fill_between(epochs, acc,val_acc,color='g',alpha=.1)
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[90]:


class_names


# In[91]:


class_names.remove('.ipynb_checkpoints')
num_classes=num_classes-1


# In[92]:


val_ds.reset()
val_ds.shuffle = False
val_ds.next()
y_prob = model.predict(val_ds)
y_pred = y_prob.argmax(axis=-1)
y_true = val_ds.labels
print(classification_report(y_true, y_pred, target_names=class_names))


# In[93]:


def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    vmax = cm.max()
    if normalize:
        title = 'Confusion matrix (normalized)'
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = [[int(j*100) for j in i ] for i in cm]
        cm =np.array(cm)
        vmax = 100
        
    plt.figure(figsize=(8,8))

    im = plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0.0, vmax=vmax)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.colorbar(im,fraction=0.046, pad=0.04)
    plt.show()


# In[94]:


val_ds.reset()
val_ds.shuffle = False
val_ds.next()
y_prob = model.predict(val_ds)
y_pred = y_prob.argmax(axis=-1)
y_true = val_ds.labels
cnf_matrix = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm=cnf_matrix, classes=class_names, title='Confusion Matrix', normalize=True)


# In[95]:


val_ds.reset()
val_ds.shuffle = True

plt.figure(figsize=(16, 16))
images = []
labels_pred = []
labels_true = []
for itr in val_ds.next():
    for i in range(25):
        if len(images) < 25:
            images.append(itr[i].astype("uint8"))
            y_proba = model.predict(np.array([itr[i]]))
            y_pred = np.argmax(y_proba,axis=1)[0]
            labels_pred.append(y_pred)
        else:
            labels_true.append(list(itr[i]).index(1))
    


# In[96]:


print(labels_true)
print(labels_pred)


# In[ ]:




