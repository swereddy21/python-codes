import pandas as pd
import numpy as np
import sys
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from pylint.lint import Run
from keras.models import load_model
from io import StringIO
from IPython import get_ipython
from configparser import ConfigParser
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from tabulate import tabulate
start_time = time.time()

ipython = get_ipython()

#############################################
def best_parameters():
    
    Hyperparameter_tuning_string = "\tRegularization Techniques applied:"
    dropout = defaultparser.get('model_param', 'dropout_list')
    dropout = dropout.split(',')
    dropout = [float(i) for i in dropout] 
    
    Hyperparameter_tuning_Variable = parser.get('Hyperparameter_tuning', 'hyperparameters')
    Hyperparameter_tuning_Variable = Hyperparameter_tuning_Variable.split(',')
    
    def Hyperparameter_tuning_matcher_Variable(x):
        for i in Hyperparameter_tuning_Variable:
            if i == x:
                return i
        else:
            return np.nan
        
    Hyperparameter_tuning_Variables = data['Type'].apply(Hyperparameter_tuning_matcher_Variable)
    Hyperparameter_tuning_Variables = np.array(Hyperparameter_tuning_Variables)
    Hyperparameter_tuning_Variables = [x for x in Hyperparameter_tuning_Variables if str(x) != 'nan']
    
    
    if len(Hyperparameter_tuning_Variables) > 0:
        for i in Hyperparameter_tuning_Variables:
            if i == 'Dropout':
                if len(set(dropout)) == 1:
                    Hyperparameter_tuning_string += '\n\t\t' + "Constant Dropouts are used"
                    Hyperparameter_tuning_string += '\n\t\t' + "Try using increasing dropouts for better accuracy"
                else:
                    Hyperparameter_tuning_string += '\n\t\t' + "Best Dropout(Increasing Dropouts) are used"
            
            
            Hyperparameter_tuning_string += '\n\t\t' + str(i)
    else:
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        if keyword20 in data2:
            Hyperparameter_tuning_string = Hyperparameter_tuning_string + '\n\t\t' + 'Dropout :'
            if len(set(dropout)) == 1:
                Hyperparameter_tuning_string += "Fixed Dropout rate is used"
                count1 = 0 
              #  Hyperparameter_tuning_string += '\n\t\t' + "Try using increasing dropout rate for better accuracy"
            else:
                Hyperparameter_tuning_string += "Best Dropout(Increasing Dropout) rate is used"
                count1 = 1
        if keyword21 in data2:
            Hyperparameter_tuning_string = Hyperparameter_tuning_string + '\n\t\t' + 'BatchNormalization ' + '\n\t\t'
            count2 = 1
        if keyword23 in data2:
            Hyperparameter_tuning_string = Hyperparameter_tuning_string + '\n\t\t' + 'Early Stopping ' + '\n\t\t'
            count3 = 1
            
        if keyword22 in data2:
            Hyperparameter_tuning_string += '\n' + 'Data Augmentation:' + '\n'
            Hyperparameter_tuning_string = Hyperparameter_tuning_string + '\n\t\t' + 'ImageDataGenerator ' + '\n\t\t'
            count4 = 1
            
        if count1 == 0 or count2 == 0 or count3 == 0 or count4 == 0:
            Hyperparameter_tuning_string1 = '\n\t\tTry using techniques like: '
            if count1 == 0:
                Hyperparameter_tuning_string1 += '\n\t\t\t' + 'Increasing Dropout Rate '
            if count2 == 0:
                Hyperparameter_tuning_string1 += '\n\t\t\t' + 'BatchNormalization  ' 
            if count3 == 0:
                Hyperparameter_tuning_string1 += '\n\t\t\t' + 'Early Stopping  '
            if count4 == 0:
                Hyperparameter_tuning_string1 += '\n\t\t\t' + 'ImageDataGenerator(Data Augmentation)  '
                
       # print(Hyperparameter_tuning_string)
                
        if count1 == 0 or count2 == 0 or count3 == 0 or count4 == 0:
            #str1 =  "\n\tParameters used while tuning the model to improve the accuracy score:"
            str1 =  "\t\tDropout - increasing rate \n\t\tBatch Normalization \n\t\tEarly Stopping - (patience = " + str(patience_value) + ", monitor = " + monitor_value +")" + "\n\t\tImageDataGenerator(Data Augmentation)\n"
               
    return Hyperparameter_tuning_string,Hyperparameter_tuning_string1,str1

#############################################
# Defining parser object and reading the Configuration file

parser = ConfigParser()
parser.read('validation.ini')

defaultparser = ConfigParser()
defaultparser.read('default.ini')

codefilename = parser.get('code_file', 'codefilename')

importtest = 'import ' + codefilename
exec(importtest)

temp_std = sys.stdout
print(temp_std)
sys.stdout = s = StringIO()


ipython.magic("whos")



GF = pd.DataFrame()
GF = s.getvalue().splitlines()

GF = pd.DataFrame(GF)
GF.reset_index(drop=True)
GF.to_csv('variables_NN.csv', index=False)

 

var_data = pd.read_csv('variables_NN.csv', sep="\t", names=['val'], header=None, skiprows=1)
var_data.val = var_data.val.replace('\s+', ' ', regex=True)
var_data = var_data.join(var_data['val'].str.split(' ', expand=True).add_prefix('val'))
var_data.drop(['val'], axis=1, inplace=True)
var_data["Description"] = var_data.iloc[:, 2:].apply(lambda x: ','.join(x.dropna()), axis=1)

var_data = var_data.rename(columns={"val0": "Variable", "val1": "Type"})
var_data = var_data.loc[:, ['Variable', 'Type', 'Description']]
var_data = var_data.iloc[1:]

var_data.to_csv('variables_NN.csv', index=False)
sys.stdout.close()

sys.stdout = temp_std


codename = codefilename + '.py'

file = open(codename, 'r')
data2 = file.read()
g = open('AI_Glassbox_Image_Classification_Model_Review.txt', 'w')
original = sys.stdout
sys.stdout = g

keyword1 = parser.get('keywords', 'keyword1')
keyword2 = parser.get('keywords', 'keyword2')
keyword3 = parser.get('keywords', 'keyword3')
keyword4 = parser.get('keywords', 'keyword4')
keyword5 = parser.get('keywords', 'keyword5')
keyword6 = parser.get('keywords', 'keyword6')
keyword7 = parser.get('keywords', 'keyword7')
keyword8 = parser.get('keywords', 'keyword8')
keyword9 = parser.get('keywords', 'keyword9')
keyword10 = parser.get('keywords', 'keyword10')
keyword11 = parser.get('keywords', 'keyword11')
keyword12 = parser.get('keywords', 'keyword12')
keyword13 = parser.get('keywords', 'keyword13')
keyword14 = parser.get('keywords', 'keyword14')
keyword15 = parser.get('keywords', 'keyword15')
keyword16 = parser.get('keywords', 'keyword16')
keyword17 = parser.get('keywords', 'keyword17')
keyword18 = parser.get('keywords', 'keyword18')
keyword19 = parser.get('keywords', 'keyword19')
keyword20 = parser.get('keywords', 'keyword20')
keyword21 = parser.get('keywords', 'keyword21')
keyword22 = parser.get('keywords', 'keyword22')
keyword23 = parser.get('keywords', 'keyword23')
keyword24 = parser.get('keywords', 'keyword24')
keyword25 = parser.get('keywords', 'keyword25')
keyword26 = parser.get('keywords', 'keyword26')
keyword27 = parser.get('keywords', 'keyword27')
keyword28 = parser.get('keywords', 'keyword28')
keyword29 = parser.get('keywords', 'keyword29')
keyword30 = parser.get('keywords', 'keyword30')
keyword31 = parser.get('keywords', 'keyword31')
keyword32 = parser.get('keywords', 'keyword32')
keyword33 = parser.get('keywords', 'keyword33')
keyword34 = parser.get('keywords', 'keyword34')
keyword35 = parser.get('keywords', 'keyword35')
keyword36 = parser.get('keywords', 'keyword36')
keyword37 = parser.get('keywords', 'keyword37')
#################################################

print("CheckPoint 1: Data Exploration:")

labels = parser.get('variables1', 'labels')
num_classes = parser.get('variables1', 'num_in each_class')
num_class = parser.get('variables1', 'num_class')
labels = 'labels = ' + codefilename + "." + labels
exec(labels)
num_classes = 'num_classes = ' + codefilename + "." + num_classes
exec(num_classes)
num_class = 'num_class = ' + codefilename + "." + num_class
exec(num_class)
print("\tLabels present in data : " + str(labels))
print("\tNumber of Images contained in each Label : " + str(num_classes))
print("\tNo eda script found in the given code")


###############################################
print("\nCheckPoint 2: Feature Engineering: ")
####################################################################################
data = pd.read_csv(parser.get('file_path', 'filename'))
  
encoders = parser.get('encoding', 'encoder_Variable')
bin_transformers = encoders.split(',')

def encoder_matcher(x):
    for i in bin_transformers:
        if i == x:
            return i
    else:
        return np.nan


encoder_matches1 = data['Type'].apply(encoder_matcher)
encoder_matches1 = np.array(encoder_matches1)
encoder_matches = [x for x in encoder_matches1 if str(x) != 'nan']

encoder_matches_string = '\tEncoder techniques check:'
if len(encoder_matches) > 0:
    for i in encoder_matches:
        encoder_matches_string += '\n\t\t' + str(i)
else:
    count = 0
    if keyword1 in data2:
        encoder_matches_string = encoder_matches_string + '\n\t\t' + 'LabelEncoder'
        count += 1
    if keyword2 in data2:
        encoder_matches_string = encoder_matches_string + '\n\t\t' + 'OneHotEncoder'
        count += 1
    if count == 0:
        encoder_matches_string += '\n\t\tNo Encoder techniques applied. Try using techniques like: LabelEncoder,OneHotEncoder'
 
print(encoder_matches_string)
####################################################################################

X_train = parser.get('variables', 'X_train')
X_test = parser.get('variables', 'X_test')
X_val = parser.get('variables', 'X_val')
y_train = parser.get('variables', 'y_train')
y_test = parser.get('variables', 'y_test')
y_val = parser.get('variables', 'y_val')


X_tra = 'X_train = ' + codefilename + "." + X_train
exec(X_tra)
X_tst = 'X_test = ' + codefilename + "." + X_test
exec(X_tst)
X_prd = 'X_val = ' + codefilename + "." + X_val
exec(X_prd)


y_tra = 'y_train = ' + codefilename + "." + y_train
exec(y_tra)
y_tst = 'y_test = ' + codefilename + "." + y_test
exec(y_tst)
y_prd = 'y_val = ' + codefilename + "." + y_val
exec(y_prd)
#####################################################################
model_api = parser.get('model', 'model_api')
model_api = model_api.split(',')

def model_matcher(x):
    for i in model_api:
        if i == x:
            return i
    else:
        return np.nan   

model_matches1 = data['Type'].apply(model_matcher)
model_matches1 = np.array(model_matches1)
model_matches = [x for x in model_matches1 if str(x) != 'nan']
model_matches_string = "\tModel used check:"
if len(model_matches) > 0:
    for i in model_matches:
        model_matches_string += '\n\t\t' + str(i)
else:
    count = 0
    if keyword6 in data2:
        model_matches_string = model_matches_string + '\n\t\t' + 'Sequential'
        count += 1
    
    if count == 0:
        model_matches_string += "\n\t\tNo Model api have applied. Try using  " \
                                        "Sequential,Functional API from keras"
print(model_matches_string)

######################################################################


holdout_techs_Variable = parser.get('holdout_techs', 'holdout_techs_Variable')
holdout_techs_Variable = holdout_techs_Variable.split(',')
holdout_techs_Type = parser.get('holdout_techs', 'holdout_techs_Type')
holdout_techs_Type = holdout_techs_Type.split(',')


def holdout_matcher_Variable(x):
    for i in holdout_techs_Variable:
        if i == x:
            return i
    else:
        return np.nan


def holdout_matcher_Type(x):
    for i in holdout_techs_Type:
        if i == x:
            return i
    else:
        return np.nan


holdouts1 = data['Type'].apply(holdout_matcher_Type)
holdouts1 = np.array(holdouts1)
holdouts2 = data['Variable'].apply(holdout_matcher_Variable)
holdouts2 = np.array(holdouts2)
holdouts1 = [x for x in holdouts1 if str(x) != 'nan']
holdouts2 = [x for x in holdouts2 if str(x) != 'nan']
holdouts_string = "\tData Splitting techniques check:"
if len(holdouts1) > 0:
    for i in holdouts1:
        holdouts_string += '\n\t\t' + str(i)
else:
    if keyword3 in data2:
        holdouts_string = holdouts_string + '\n\t\t' + 'Kfold'
    if keyword4 in data2:
        holdouts_string = holdouts_string + '\n\t\t' + 'Bootstrap'

if len(holdouts2) > 0:
    for i in holdouts2:
        holdouts_string += '\n\t\t' + str(i)
else:
    if keyword5 in data2:
        holdouts_string = holdouts_string + '\n\t\t' + 'train_test_split'

if (len(holdouts1) == 0) and (len(holdouts2) == 0):
    if (keyword3 not in data2) and (keyword4 not in data2) and (keyword5 not in data2):
        holdouts_string += "\n\t\tNo Data splitting techniques were applied. Try using techniques like: Train Test " \
                            "Split, Kfold Cross Validation, Bootstrap "
print(holdouts_string)
############################################################
result_layer_df = pd.DataFrame(data=None,columns=['Layer', 'used'])
print("\nCheckPoint 3: CNN and Metrices:")
print("\tDifferent Layers Used:")
model_layers = parser.get('convolution_layer', 'convolution')
model_layers = model_layers.split(',')

def model_layers_matcher(x):
    for i in model_layers:
        if i == x:
            return i
    else:
        return np.nan

model_layers_matches1 = data['Type'].apply(model_layers_matcher)
model_layers_matches1 = np.array(model_layers_matches1)
model_layers_matches = [x for x in model_layers_matches1 if str(x) != 'nan']
#model_layers_matches_string = "\tDiffernet Layers Used:"
model_layers_matches_string = ""
if len(model_layers_matches) > 0:
    for i in model_layers_matches:
       model_layers_matches_string += str(i)
      # print(model_layers_matches_string)
else:
    count = 0
    if keyword17 in data2:
        model_layers_matches_string = model_layers_matches_string + 'Conv2D'
        count += 1
    if keyword27 in data2:
        model_layers_matches_string = model_layers_matches_string + 'Conv2DTranspose'
        count += 1
    if count == 0:
        model_layers_matches_string += "No Convolution layers used. Try using layers like: " \
                                        "Conv2D, Conv2DTranspose "
#print(model_layers_matches_string)
df_layer = pd.DataFrame({'Layer': ['Convolution Layers'],
                         'used': [model_layers_matches_string]})
result_layer_df  = result_layer_df.append(df_layer, ignore_index=True)
#############################################
model_layers1 = parser.get('core_layer', 'core_Layers')
model_layers1 = model_layers1.split(',')

def model_layers_matcher1(x):
    for i in model_layers1:
        if i == x:
            return i
    else:
        return np.nan

model_layers_matches1_1 = data['Type'].apply(model_layers_matcher1)
model_layers_matches1_1 = np.array(model_layers_matches1_1)
model_layers_matches1 = [x for x in model_layers_matches1_1 if str(x) != 'nan']
#model_layers_matches_string1 = "\t\tCore Layers:"
model_layers_matches_string1 = ""
if len(model_layers_matches1) > 0:
    for i in model_layers_matches1:
       model_layers_matches_string1 += str(i)
      # print(model_layers_matches_string)
else:
    count = 0
    if keyword16 in data2:
        model_layers_matches_string1 = model_layers_matches_string1 + 'Dense'
        count += 1
    if keyword28 in data2:
        model_layers_matches_string1 = model_layers_matches_string1 + 'Embedding'
        count += 1
    if keyword29 in data2:
        model_layers_matches_string1 = model_layers_matches_string1 + 'Masking'
        count += 1
    if count == 0:
        model_layers_matches_string1 += "No core layers used. Try using layers like: " \
                                        "Dense, Embedding "
#print(model_layers_matches_string1)
df_layer = pd.DataFrame({'Layer': ['Core Layers'],
                         'used': [model_layers_matches_string1]})
result_layer_df  = result_layer_df.append(df_layer, ignore_index=True) 
   
#############################################################
model_layers2 = parser.get('pooling_layer', 'pooling_Layers')
model_layers2 = model_layers2.split(',')

def model_layers_matcher2(x):
    for i in model_layers2:
        if i == x:
            return i
    else:
        return np.nan

model_layers_matches1_2 = data['Type'].apply(model_layers_matcher2)
model_layers_matches1_2 = np.array(model_layers_matches1_2)
model_layers_matches2 = [x for x in model_layers_matches1_2 if str(x) != 'nan']
#model_layers_matches_string2 = "\t\tPooling Layers:"
model_layers_matches_string2= ""
if len(model_layers_matches2) > 0:
    for i in model_layers_matches2:
       model_layers_matches_string2 += str(i)
      # print(model_layers_matches_string)
else:
    count = 0
    if keyword19 in data2:
        model_layers_matches_string2 = model_layers_matches_string2 + 'MaxPooling2D'
        count += 1
    if keyword18 in data2:
        model_layers_matches_string2 = model_layers_matches_string2 + 'AveragePooling2D'
        count += 1
    if count == 0:
        model_layers_matches_string2 += "No pooling layers used. Try using layers like: " \
                                        "AveragePooling2D, MaxPooling2D"
#print(model_layers_matches_string2)
df_layer = pd.DataFrame({'Layer': ['Pooling Layers'],
                         'used': [model_layers_matches_string2]})
result_layer_df  = result_layer_df.append(df_layer, ignore_index=True) 
    
########################################################
model_layers3 = parser.get('Regularization_layer', 'regularization')
model_layers3 = model_layers3.split(',')

def model_layers_matcher3(x):
    for i in model_layers3:
        if i == x:
            return i
    else:
        return np.nan

model_layers_matches1_3 = data['Type'].apply(model_layers_matcher3)
model_layers_matches1_3 = np.array(model_layers_matches1_3)
model_layers_matches3 = [x for x in model_layers_matches1_3 if str(x) != 'nan']
#model_layers_matches_string3 = "\t\tRegularization Layers:"
model_layers_matches_string3 = ""
if len(model_layers_matches3) > 0:
    for i in model_layers_matches3:
       model_layers_matches_string3 += str(i)
      # print(model_layers_matches_string)
else:
    count = 0
    if keyword25 in data2:
        model_layers_matches_string3 = model_layers_matches_string3 + 'Dropout'
        count += 1
    if keyword30 in data2:
        model_layers_matches_string3 = model_layers_matches_string3 +'GaussianDropout'
        count += 1
    if count == 0:
        model_layers_matches_string3 += "No Regularization layers used. Try using layers like: " \
                                         "Dropout, GaussianDropout"
#print(model_layers_matches_string3)
df_layer = pd.DataFrame({'Layer': ['Regularization Layers'],
                         'used': [model_layers_matches_string3]})
result_layer_df  = result_layer_df.append(df_layer, ignore_index=True) 

#########################################################
model_layers4 = parser.get('Normalization_layer', 'normalization')
model_layers4 = model_layers4.split(',')

def model_layers_matcher4(x):
    for i in model_layers4:
        if i == x:
            return i
    else:
        return np.nan

model_layers_matches1_4 = data['Type'].apply(model_layers_matcher4)
model_layers_matches1_4 = np.array(model_layers_matches1_4)
model_layers_matches4 = [x for x in model_layers_matches1_4 if str(x) != 'nan']
#model_layers_matches_string4 = "\t\tNormalization Layers:"
model_layers_matches_string4= ""
if len(model_layers_matches4) > 0:
    for i in model_layers_matches4:
       model_layers_matches_string4 += str(i)
      # print(model_layers_matches_string)
else:
    count = 0
    if keyword21 in data2:
        model_layers_matches_string4 = model_layers_matches_string4 + 'BatchNormalization'
        count += 1
    if keyword31 in data2:
        model_layers_matches_string4 = model_layers_matches_string4 + 'LayerNormalization'
        count += 1
    if count == 0:
        model_layers_matches_string4 += "No Normalization layers used. Try using layers like: " \
                                        "BatchNormalization, LayerNormalization "
#print(model_layers_matches_string4)
df_layer = pd.DataFrame({'Layer': ['Normalization Layers'],
                         'used': [model_layers_matches_string4]})
result_layer_df  = result_layer_df.append(df_layer, ignore_index=True) 

#########################################################
model_layers5 = parser.get('reshaping_layer', 'reshaping_layer')
model_layers5 = model_layers5.split(',')

def model_layers_matcher5(x):
    for i in model_layers5:
        if i == x:
            return i
    else:
        return np.nan

model_layers_matches1_5 = data['Type'].apply(model_layers_matcher5)
model_layers_matches1_5 = np.array(model_layers_matches1_5)
model_layers_matches5 = [x for x in model_layers_matches1_5 if str(x) != 'nan']
#model_layers_matches_string5 = "\t\tReshaping Layer:"
model_layers_matches_string5 = ""
if len(model_layers_matches5) > 0:
    for i in model_layers_matches5:
       model_layers_matches_string5 += str(i)
      # print(model_layers_matches_string)
else:
    count = 0
    if keyword24 in data2:
        model_layers_matches_string5 = model_layers_matches_string5 +  'Flatten'
        count += 1
    if keyword32 in data2:
        model_layers_matches_string5 = model_layers_matches_string5 + 'Reshape'
        count += 1
    if count == 0:
        model_layers_matches_string5 += "No Reshaping layers used. Try using layers like:" \
                                        "Reshape, Flatten "
#print(model_layers_matches_string5)
df_layer = pd.DataFrame({'Layer': ['Reshaping Layers'],
                         'used': [model_layers_matches_string5]})
result_layer_df  = result_layer_df.append(df_layer, ignore_index=True)
 

pdtabulate=lambda df:tabulate(result_layer_df,headers='keys',tablefmt='psql')

print(pdtabulate(result_layer_df))
######################################################
activation_techs = parser.get('activation_techniques', 'activation_techs')
activation_techs = activation_techs.split(',')

def activation_matcher(x):
    for i in activation_techs:
        if i == x:
            return i
    else:
        return np.nan

activation_matches1 = data['Type'].apply(activation_matcher)
activation_matches1 = np.array(activation_matches1)
activation_matches = [x for x in activation_matches1 if str(x) != 'nan']
activation_matches_string = "\tActivation techniques applied:"
if len(activation_matches) > 0:
    for i in activation_matches:
       activation_matches_string += '\n\t\t' + str(i)
       
else:
    count = 0
    if keyword11 in data2:
        activation_matches_string = activation_matches_string + '\n\t\t' + 'relu'
        count += 1
    if keyword12 in data2:
        activation_matches_string = activation_matches_string + '\n\t\t' + 'sigmoid'
        count += 1
    if keyword13 in data2:
        activation_matches_string = activation_matches_string + '\n\t\t' + 'softmax'
        count += 1
    if keyword14 in data2:
        activation_matches_string = activation_matches_string + '\n\t\t' + 'Tanh'
        count += 1
    if keyword15 in data2:
        activation_matches_string = activation_matches_string + '\n\t\t' + 'softsign'
        count += 1
    if count == 0:
        activation_matches_string += "\n\t\tNo Activation functions applied. Try using techniques like: " \
                                      "relu, softmax "

print(activation_matches_string)
#######################################################
img_aug = parser.get('data_augmentation', 'aug_techs')
img_aug = img_aug.split(',')

def img_aug_matcher(x):
    for i in img_aug:
        if i == x:
            return i
    else:
        return np.nan

img_aug_matches1 = data['Type'].apply(img_aug_matcher)
img_aug_matches1 = np.array(img_aug_matches1)
img_aug_matches = [x for x in img_aug_matches1 if str(x) != 'nan']
img_aug_matches_string = "\tData Augumentation Used:"
if len(img_aug_matches) > 0:
    for i in img_aug_matches:
       img_aug_matches_string += '\n\t\t' + str(i)
      # print(model_layers_matches_string)
else:
    count = 0
    if keyword22 in data2:
        img_aug_matches__string = img_aug_matches_string + '\n\t\t' + 'ImageDataGenerator'
        count += 1
    if count == 0:
        img_aug_matches_string += "\n\t\tNo Augmentation techniques used. Try using techniques like: " \
                                   " ImageDataGenerator "
print(img_aug_matches_string)
##########################################################
model_metric = parser.get('model_metric_techniques', 'model_metric_techs')
model_metric_techs = model_metric.split(',')

def model_metric_matcher(x):
    for i in model_metric_techs:
        if i == x:
            return i
    else:
        return np.nan 
    
CNN_metric_matches1 = data['Type'].apply(model_metric_matcher)
CNN_metric_matches1 = np.array(CNN_metric_matches1)
CNN_metric_matches = [x for x in CNN_metric_matches1 if str(x) != 'nan']

#print(CNN_metric_matches)
CNN_metric_matches_string = "\tCNN Metrics applied:"
if len(CNN_metric_matches) > 0:
    for i in CNN_metric_matches:
        CNN_metric_matches_string += '\n\t\t' + str(i)
        
else:
    count = 0
    if keyword7 in data2:
        CNN_metric_matches_string = CNN_metric_matches_string + '\n\t\t' + 'accuracy_score'
        count += 1
    if keyword8 in data2:
        CNN_metric_matches_string = CNN_metric_matches_string + '\n\t\t' + 'precision_score'
        count += 1
    if keyword9 in data2:
        CNN_metric_matches_string = CNN_metric_matches_string + '\n\t\t' + 'recall_score'
        count += 1
    if keyword10 in data2:
        CNN_metric_matches_string = CNN_metric_matches_string + '\n\t\t' + 'classification_report'
        count += 1
    if keyword24 in data2:
        CNN_metric_matches_string = CNN_metric_matches_string + '\n\t\t' + 'f1_score'
        count += 1
    if count == 0:
        CNN_metric_matches_string += "\n\t\tNo Metrics have applied. Try using accuracy_score"
print(CNN_metric_matches_string)
optimizer_var = defaultparser.get('model_param','optimizer')
loss_var = defaultparser.get('model_param','loss')
print("\t Model Parameters Used:")
print("\t\tLoss function :",loss_var)
print("\t\tOptimizer :", optimizer_var)
epochs_var = defaultparser.getint('model_param','epochs')
batchsize_var = defaultparser.getint('model_param','batch_size')
print("\t\tEpochs :",epochs_var)
print("\t\tBatch size :",batchsize_var)

########################################################################

result_df = pd.DataFrame(data=None,
                         columns=['Model', 'Train Accuracy', 'Test Accuracy', 'val Accuracy','Train F1', 'Test F1','val F1',
                                  'Train Precision', 'Test Precision', 'val Precision','Train Recall', 'Test Recall','val Recall',
                                  'OD_acc_test','OD_acc_val','OD_f1_test','OD_f1_val'])
       

if 'Sequential' in model_matches:
    modelused = data[data['Type'] == 'Sequential']['Variable']

    modelused = pd.DataFrame(data=modelused)
    modelused = modelused.iloc[0]
    modelused = str(modelused[0])
    model_summary = "model_summary = " + codefilename + "." + modelused + ".summary()"
    print("\n\tBaseline Model Summary: ")
    exec(model_summary)
    print('\n')
    predictions = "predictions = " + codefilename + "." + modelused
        
    exec(predictions)
    
    y_train_pred = predictions.predict_classes(X_train)
    y_train_actual = np.argmax(y_train, axis = 1)
    
    train_acc_baseline = accuracy_score(y_train_pred,y_train_actual)
    train_f1_baseline = f1_score(y_train_actual, y_train_pred,average='weighted')
    train_precision_baseline = precision_score(y_train_actual, y_train_pred,average='weighted')
    train_recall_baseline = recall_score(y_train_actual, y_train_pred,average='weighted')
    
    print("\tMetrics for train data using Baseline model: ")
       
    print("\t\tAccuracy_score: %.3f" % train_acc_baseline)
    print("\t\tF1_score: %.3f" % train_f1_baseline)
    print("\t\tPrecision_score: %.3f" % train_precision_baseline)
    print("\t\tRecall_score: %.3f" % train_recall_baseline)
           
    y_test_pred = predictions.predict_classes(X_test)
    y_test_actual = np.argmax(y_test, axis = 1)
    
    test_acc_baseline = accuracy_score(y_test_pred, y_test_actual)
    test_f1_baseline = f1_score(y_test_actual, y_test_pred,average='weighted')
    test_precision_baseline = precision_score(y_test_actual, y_test_pred,average='weighted')
    test_recall_baseline = recall_score(y_test_actual, y_test_pred,average='weighted')
    
    print("\tMetrics for test data using Baseline model: ")
       
    print("\t\tAccuracy_score: %.3f" % test_acc_baseline)
    print("\t\tF1_score: %.3f" % test_f1_baseline)
    print("\t\tPrecision_score: %.3f" % test_precision_baseline)
    print("\t\tRecall_score: %.3f" % test_recall_baseline)
           


    df_baseline = pd.DataFrame({'Model': ['baseline model'],
                                'Train Accuracy': [train_acc_baseline], 'Test Accuracy': [test_acc_baseline],
                                'Train F1': [train_f1_baseline], 'Test F1': [test_f1_baseline],
                                'Train Precision': [train_precision_baseline], 'Test Precision': [test_precision_baseline],
                                'Train Recall': [train_recall_baseline],'Test Recall' :[test_recall_baseline]})
    
    result_df = result_df.append(df_baseline, ignore_index=True)
    #print(result_df)


################################################################
# HYPERPARAMETR_Check = True

hyperparameter_Check = parser.get('hyperparameter_checks', 'hyperparameter_check')
overfitting_check = float(parser.get('overfitting_checks', 'overfitting_value'))

if hyperparameter_Check == 'True':
    print("\nCheckPoint 4: Hyperparameter Tuning: ")


    #### Best Parameters ####
    tunedmodel = Sequential()
    tunedmodel.add(Conv2D(32, kernel_size=(5, 5), input_shape=X_train.shape[1:], activation ="relu"))
    tunedmodel.add(BatchNormalization())
    tunedmodel.add(MaxPooling2D(pool_size=(2, 2)))
    tunedmodel.add(Dropout(0.1))
        
    tunedmodel.add(Conv2D(64, kernel_size=(3, 3),activation ="relu"))
    tunedmodel.add(BatchNormalization())
    tunedmodel.add(MaxPooling2D(pool_size=(2, 2)))
    tunedmodel.add(Dropout(0.2))
        
    tunedmodel.add(Conv2D(128, kernel_size=(3, 3),activation ="relu"))
    tunedmodel.add(BatchNormalization())
    tunedmodel.add(MaxPooling2D(pool_size=(2, 2)))
    tunedmodel.add(Dropout(0.3))
        
    tunedmodel.add(Flatten())
    tunedmodel.add(Dense(128,activation='relu'))
    tunedmodel.add(BatchNormalization())
    tunedmodel.add(Dropout(0.4))
    tunedmodel.add(Dense(num_class, activation='softmax'))
    #print(tunedmodel.summary())
      
    tunedmodel.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    aug = ImageDataGenerator(width_shift_range=0.1, 
                             height_shift_range=0.1, 
                             horizontal_flip=True)
    global patience_value
    global monitor_value
    patience_value = 10
    monitor_value = 'val_loss'
    es = EarlyStopping(monitor=monitor_value, mode='min', verbose=0, patience=patience_value)
    mc = ModelCheckpoint('tuned_model.h5', monitor='val_accuracy', mode='max', verbose=0, save_best_only=True)
    
    
    tunedmodel.fit(x=aug.flow(X_train, y_train, batch_size=32),
                   steps_per_epoch=len(X_train) // 32,
                   validation_data = (X_test, y_test),
                   callbacks=[es, mc],
                   epochs=50,
                   verbose = 0)
    # load the saved model
    saved_model = load_model('tuned_model.h5')
    
    y_train_pred = saved_model.predict_classes(X_train)
    y_train_actual = np.argmax(y_train, axis = 1)
    train_acc_best = accuracy_score(y_train_pred,y_train_actual)
    train_f1_best = f1_score(y_train_actual, y_train_pred,average='weighted')
    train_precision_best = precision_score(y_train_actual, y_train_pred,average='weighted')
    train_recall_best = recall_score(y_train_actual, y_train_pred,average='weighted')
  
    y_test_pred = saved_model.predict_classes(X_test)
    y_test_actual = np.argmax(y_test, axis = 1)
    test_acc_best = accuracy_score(y_test_pred,y_test_actual)
    test_f1_best = f1_score(y_test_actual, y_test_pred,average='weighted')
    test_precision_best = precision_score(y_test_actual, y_test_pred,average='weighted')
    test_recall_best = recall_score(y_test_actual, y_test_pred,average='weighted')
    
    
    df_tuned = pd.DataFrame({'Model': ['tuned model'],
                             'Train Accuracy': [train_acc_best], 'Test Accuracy': [test_acc_best],
                             'Train F1': [train_f1_best], 'Test F1': [test_f1_best],
                             'Train Precision': [train_precision_best], 'Test Precision': [test_precision_best],
                             'Train Recall': [train_recall_best],'Test Recall' :[test_recall_best]})
    result_df = result_df.append(df_tuned, ignore_index=True)
    #print('the accuracy obtained on the best test set is:', test_acc_best)
    #print(classification_report(y_test_actual, y_test_pred))
    param1,param2,param3 = best_parameters()
    print(param1)
    if float(train_acc_baseline - test_acc_baseline) > overfitting_check:
        print("\n\tThe baseline model's performance metrics values show the model is tending towards overfitting")
        if (train_acc_best - test_acc_best) < overfitting_check:
            print("\n\tConsider tuning the model hyper parameters.Below parameters can be used to tune hyper parameters in the base line model to avoid over fitting:")
           # print(param2)
            print(param3)
            print('\tTuned Model Summary :')
            tunedmodel.summary()
            print('\n')
            print("\tMetrics for train data using tuned model: ")
            print('\t\tAccuracy score: %.3f' % train_acc_best)
            print('\t\tF1 score: %.3f' % train_f1_best)
            print('\t\tPrecision: %.3f' % train_precision_best)
            print('\t\tRecall: %.3f' % train_recall_best)
                               
            print("\tMetrics for test data using tuned model: ")
            print('\t\tAccuracy score: %.3f' % test_acc_best)
            print('\t\tF1 score: %.3f' % test_f1_best)
            print('\t\tPrecision: %.3f' % test_precision_best)
            print('\t\tRecall: %.3f' % test_recall_best)
        else:
            print("\n\tPlease tweak the hyper parameters and re-train model")
            
    elif test_acc_best > test_acc_baseline:
        print("\n\tConsider tuning the model hyper parameters. Below parameters can be used to tune hyper parameters in the base line model to improve the accuracy score:")

        print(param3)
        print('\tTuned Model Summary :')
        tunedmodel.summary()
        print('\n')
        print("\tMetrics for train data using tuned model: ")
        print('\t\tAccuracy score: %.3f' % train_acc_best)
        print('\t\tF1 score: %.3f' % train_f1_best)
        print('\t\tPrecision: %.3f' % train_precision_best)
        print('\t\tRecall: %.3f' % train_recall_best)
                               
        print("\tMetrics for test data using tuned model: ")
        print('\t\tAccuracy score: %.3f' % test_acc_best)
        print('\t\tF1 score: %.3f' % test_f1_best)
        print('\t\tPrecision: %.3f' % test_precision_best)
        print('\t\tRecall: %.3f' % test_recall_best)
        
    else:
        print("\n\tThe baseline model has been trained using tuned parameters")
    ########
    
    print('\n\n\tVarious metrics for val data using different algorithms')
    
     
    
    y_val_pred = predictions.predict_classes(X_val)
    y_val_actual = np.argmax(y_val, axis = 1)
    
     
    
    val_acc_baseline = accuracy_score(y_val_pred,y_val_actual)
    val_f1_baseline = f1_score(y_val_actual, y_val_pred,average='weighted')
    val_precision_baseline = precision_score(y_val_actual, y_val_pred,average='weighted')
    val_recall_baseline = recall_score(y_val_actual, y_val_pred,average='weighted')
    
     
    
    print("\tUsing Baseline model: ")
    print("\t\tAccuracy score: %.3f" % val_acc_baseline)
    print("\t\tF1 score: %.3f" % val_f1_baseline)
    print("\t\tPrecision: %.3f" % val_precision_baseline)
    print("\t\tRecall: %.3f" % val_recall_baseline)
        
    result_df.loc[result_df['Model'] == 'baseline model', ['val Accuracy', 'val F1', 'val Precision','val Recall']] = val_acc_baseline, val_f1_baseline, val_precision_baseline, val_recall_baseline
                    
    y_val_pred = saved_model.predict_classes(X_val)
    y_val_actual = np.argmax(y_val, axis = 1)
    val_acc_best = accuracy_score(y_val_pred,y_val_actual)
    val_f1_best = f1_score(y_val_actual, y_val_pred,average='weighted')
    val_precision_best = precision_score(y_val_actual, y_val_pred,average='weighted')
    val_recall_best = recall_score(y_val_actual, y_val_pred,average='weighted')
                    
    print("\tUsing Tuned model: ")
    print("\t\tAccuracy score: %.3f" % val_acc_best)
    print("\t\tF1 score: %.3f" % val_f1_best)
    print("\t\tPrecision: %.3f" % val_precision_best)
    print("\t\tRecall: %.3f" % val_recall_best)
    result_df.loc[result_df['Model'] == 'tuned model', ['val Accuracy', 'val F1', 'val Precision','val Recall']] = val_acc_best, val_f1_best, val_precision_best, val_recall_best
    
    
    print('\n\n\tOverfitting degree for baseline and tuned model')
    
    print('\tBaseline model Accuracy: ')
    
    abs_od_acc_test_baseline = train_acc_baseline - test_acc_baseline
    per_od_acc_test_baseline = ((test_acc_baseline - train_acc_baseline) / train_acc_baseline) * 100
    print('\t\tFrom train to test: %.3f' % abs_od_acc_test_baseline)
    print('\t\tFrom train to test(perc change): %.3f' % per_od_acc_test_baseline)
                   
                   
                   
    abs_od_acc_val_baseline = test_acc_baseline - val_acc_baseline
    per_od_acc_val_baseline = ((val_acc_baseline - test_acc_baseline) / test_acc_baseline) * 100
    print('\t\tFrom test to val: %.3f' % abs_od_acc_val_baseline)
    print('\t\tFrom test to val(perc change): %.3f' % per_od_acc_val_baseline)
    
    print('\tBaseline model F1 Score: ')
    
    abs_od_f1_test_baseline = train_f1_baseline - test_f1_baseline
    per_od_f1_test_baseline = ((test_f1_baseline - train_f1_baseline) / train_f1_baseline) * 100
    print('\t\tFrom train to test: %.3f' % abs_od_f1_test_baseline)
    print('\t\tFrom train to test(perc change): %.3f' % per_od_f1_test_baseline)
                   
                  
    abs_od_f1_val_baseline = test_f1_baseline - val_f1_baseline
    per_od_f1_val_baseline = ((val_f1_baseline - test_f1_baseline) / test_f1_baseline) * 100
    print('\t\tFrom test to val: %.3f' % abs_od_f1_val_baseline)
    print('\t\tFrom test to val(perc change): %.3f' % per_od_f1_val_baseline)
                   
    result_df.loc[result_df['Model'] == 'baseline model', ['OD_acc_test', 'OD_acc_val', 'OD_f1_test', 'OD_f1_val']] = abs_od_acc_test_baseline, abs_od_acc_val_baseline, abs_od_f1_test_baseline, abs_od_f1_val_baseline
    #print(result_df)
    
    print('\tTuned model Accuracy: ')
    abs_od_acc_test_best = train_acc_best - test_acc_best
    per_od_acc_test_best = ((test_acc_best - train_acc_best) / train_acc_best) * 100
    print('\t\tFrom train to test: %.3f' % abs_od_acc_test_best)
    print('\t\tFrom train to test(perc change): %.3f' % per_od_acc_test_best)
    
    abs_od_acc_val_best = test_acc_best - val_acc_best
    per_od_acc_val_best = ((val_acc_best - test_acc_best) / test_acc_best) * 100
    print('\t\tFrom test to val: %.3f' % abs_od_acc_val_best)
    print('\t\tFrom test to val(perc change): %.3f' % per_od_acc_val_best)
               
    print('\tTuned model F1 Score: ')
    
    abs_od_f1_test_best = train_f1_best - test_f1_best
    per_od_f1_test_best = ((test_f1_best - train_f1_best) / train_f1_best) * 100
    print('\t\tFrom train to test: %.3f' % abs_od_f1_test_best)
    print('\t\tFrom train to test(perc change): %.3f' % per_od_f1_test_best)
                   
                  
    abs_od_f1_val_best = test_f1_best - val_f1_best
    per_od_f1_val_best = ((val_f1_best - test_f1_best) / test_f1_best) * 100
    print('\t\tFrom test to val: %.3f' % abs_od_f1_val_best)
    print('\t\tFrom test to val(perc change): %.3f' % per_od_f1_val_best)
    result_df.loc[result_df['Model'] == 'tuned model', ['OD_acc_test', 'OD_acc_val', 'OD_f1_test', 'OD_f1_val']] = abs_od_acc_test_best,abs_od_acc_val_best, abs_od_f1_test_best, abs_od_f1_val_best

print("\nCheckPoint 5: ")
retrain_threshold = parser.getint('retrain_metrics','retrain_threshold')
if(abs(((test_f1_baseline - val_f1_baseline)/test_f1_baseline) * 100) > retrain_threshold) :
    print("\tConsidering model's test and validation metrics stats - There is a significant difference between the F1 score of Test and Validation dataset. Hence, the model needs to be retrained")
else:
    print("\tConsidering model's test and validation metrics stats - No retraining required")
pd.set_option('display.max_columns', None)
print("\nModel Metrics Summary :\n")
print(result_df)    
        
################################################################

sys.stdout.flush()  # This will go to stdout and the file out.txt

sys.stdout = original


###########################################################
f = open('AI_Glassbox_Image_Classification_Code_Review.txt', 'w')
original1 = sys.stdout
sys.stdout = f
results = Run([codename], do_exit=False)
print(results.linter.stats)
sys.stdout.flush()  # This will go to stdout and the file out.txt

sys.stdout = original1
with open("AI_Glassbox_Image_Classification_Code_Review.txt", "r") as f1:
    data1 = f1.readlines()
    sys.stdout.flush()
    last_line = data1[-5]
    print(last_line)
    print("AI Glassbox model review Report in AI_Glassbox_Image_Classification_Model_Review.txt file")
    print("AI Glassbox code review Report in AI_Glassbox_Image_Classification_Code_Review.txt file")
    f1.close()

print("Execution time {} seconds ".format(np.round(time.time() - start_time, 2)))


##########################################################