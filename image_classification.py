import os
import numpy as np
from IPython.display import Image, display
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from sklearn.metrics import classification_report,accuracy_score
from configparser import ConfigParser
#from keras.utils.vis_utils import plot_model
import time

start_time = time.time()


parser = ConfigParser()
parser.read('default.ini')
folder_name = parser.get('folder_name', 'folder')
print(folder_name)
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()
    
def load_image(filename):
    im = mpimg.imread(filename,0)
    plt.imshow(im)
    plt.show()
    
    img = load_img(filename, target_size=(32, 32))
	# convert to array
    img = img_to_array(img)
	# reshape into a single sample with 3 channels
    img = img.reshape(1, 32, 32, 3)
	# prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

     
labels = os.listdir(folder_name+'//')
print(labels)
num_classes = len(labels)
num = []
for label in labels:
    path = folder_name+'/{0}/'.format(label)
    folder_data = os.listdir(path)
    k = 0
    print('\n', label.upper())
    for image_path in folder_data:
        if k < 2:
            display(Image(path+image_path))
        k = k+1
    num.append(k)
    print('there are ', k,' images in ', label, 'class')
plt.figure(figsize = (6,6))
plt.bar(labels, num)
plt.title('NUMBER OF IMAGES CONTAINED IN EACH CLASS')
plt.xlabel('classes')
plt.ylabel('count')
plt.show()

x_data =[]
y_data = []

for label in labels:
    path = folder_name+'/{0}/'.format(label)
    folder_data = os.listdir(path)
    for image_path in folder_data:
        image = cv2.imread(path+image_path)
        image_resized = cv2.resize(image, (32,32))
        x_data.append(np.array(image_resized))
        y_data.append(label)
        
x_data = np.array(x_data)
y_data = np.array(y_data)
print('the shape of X is: ', x_data.shape, 'and that of Y is: ', y_data.shape)

#lets shuffle all the data we have:
r = np.arange(x_data.shape[0])
np.random.seed(42)
np.random.shuffle(r)
X = x_data[r]
Y = y_data[r]
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20)
test_per = int(parser.get('test_percentage','test_per'))
split = test_per/100
val_per = int(parser.get('val_percentage','val_per'))
split1 = val_per/100
X_1, X_val, Y_1, Y_val = train_test_split(X, Y, test_size=split1, random_state=2)
X_train, X_test, Y_train, Y_test = train_test_split(X_1, Y_1, test_size=split, random_state=42)

print('the shape of training data X_train is: ', X_train.shape, 'and that of Y_train is: ', Y_train.shape)
print('the shape of testing data X_test is: ', X_test.shape, 'and that of Y_test is: ', Y_test.shape)
print('the shape of testing data X_val is: ', X_val.shape, 'and that of Y_test is: ', Y_val.shape)

#standardizing the input data
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255
X_val = X_val.astype('float32')/255

#converting the y_data into categorical:
le = LabelEncoder()
y_encode_train = le.fit_transform(Y_train)
y_encode_test = le.transform(Y_test)
y_encode_val = le.transform(Y_val)

Y_train = to_categorical(y_encode_train)
Y_test = to_categorical(y_encode_test)
Y_val = to_categorical(y_encode_val)

filters = parser.get('model_param', 'no_of_filters')
filters = filters.split(',')
filters = [int(i) for i in filters] 

dropout = parser.get('model_param', 'dropout_list')
dropout = dropout.split(',')
dropout = [float(i) for i in dropout] 

kernelsize = parser.get('model_param', 'kernelsize_list')
kernelsize = kernelsize.split(',')
kernelsize = [int(i) for i in kernelsize] 

poolsize = parser.get('model_param', 'poolsize_list')
poolsize = poolsize.split(',')
poolsize = [int(i) for i in poolsize] 


model = Sequential()    
model.add(Conv2D(filters[0], kernel_size=kernelsize[0], input_shape=X_train.shape[1:], activation ="relu"))
model.add(MaxPooling2D(pool_size=poolsize[0]))
model.add(Dropout(dropout[0]))
    
model.add(Conv2D(filters[1], kernel_size=kernelsize[1],activation ="relu"))
model.add(MaxPooling2D(pool_size=poolsize[1]))
model.add(Dropout(dropout[1]))

model.add(Conv2D(filters[2], kernel_size=kernelsize[2],activation ="relu"))
model.add(MaxPooling2D(pool_size=poolsize[2]))
model.add(Dropout(dropout[2]))

model.add(Flatten())
model.add(Dense(filters[2],activation='relu'))
model.add(Dropout(dropout[2]))
model.add(Dense(num_classes, activation='softmax'))
print(model.summary())
#plot_model(model,to_file = 'model.png', show_shapes=True,show_layer_names=True)

optimizer_var = parser.get('model_param','optimizer')
loss_var = parser.get('model_param','loss')
model.compile(loss=loss_var,
              optimizer=optimizer_var,
              metrics=['accuracy'])

epoch_value = parser.getint('model_param','epochs')
batch_size_value = parser.getint('model_param','batch_size')

history = model.fit(X_train, Y_train,
                    batch_size=batch_size_value,
                    epochs=epoch_value,
                    validation_data = (X_test, Y_test),
                    verbose=1)

summarize_diagnostics(history)
################
_, train_acc = model.evaluate(X_train, Y_train, verbose=0)
_, test_acc = model.evaluate(X_test, Y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

Y_pred_val = model.predict_classes(X_val)
#converting over Y test to actual labels.
Y_val_actual = np.argmax(Y_val,axis = 1)
print('the accuracy obtained on the validation set is:', accuracy_score(Y_pred_val,Y_val_actual))
print(classification_report(Y_val_actual, Y_pred_val))

#################

img = load_image('sample_image1.jpg')
result = model.predict_classes(img)
print(result)
print(labels[result[0]])

img1 = load_image('sample_image2.png')
result1 = model.predict_classes(img1)
print(result1)
print(labels[result1[0]])

print("Execution time {} seconds ".format(np.round(time.time() - start_time, 2)))

