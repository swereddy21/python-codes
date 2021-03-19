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
import matplotlib.pyplot as plt
start_time = time.time()

ipython = get_ipython()

#############################################
def best_parameters():
   
    Hyperparameter_tuning_string = """<html><p>&emsp;&emsp;Regularization Techniques applied:</p><html>"""
   
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
                    Hyperparameter_tuning_string += """<html><p>&emsp;&emsp;&emsp;&emsp;Fixed Dropout rate is used</p><html>"""
                    Hyperparameter_tuning_string += """<html><p>&emsp;&emsp;&emsp;&emsp;Try using increasing dropouts for better accuracy</p><html>"""
                else:
                    Hyperparameter_tuning_string += """<html><p>&emsp;&emsp;&emsp;&emsp;Best Dropout(Increasing Dropouts) are used</p><html>"""
           
           
            Hyperparameter_tuning_string += '\n\t\t' + str(i)
    else:
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        if keyword20 in data2:
           # Hyperparameter_tuning_string = Hyperparameter_tuning_string +  """<html><p>&emsp;&emsp;&emsp;&emsp;Dropout :</p><html>"""
            if len(set(dropout)) == 1:
                Hyperparameter_tuning_string = Hyperparameter_tuning_string +  """<html><p>&emsp;&emsp;&emsp;&emsp;Dropout : Fixed Dropout rate is used</p></html>"""
                count1 = 0
              #  Hyperparameter_tuning_string += '\n\t\t' + "Try using increasing dropout rate for better accuracy"
            else:
                Hyperparameter_tuning_string += """<html><p>&emsp;&emsp;&emsp;&emsp;Dropout : Best Dropout(Increasing Dropout) rate is used</p><html>"""
                count1 = 1
        if keyword21 in data2:
            Hyperparameter_tuning_string = Hyperparameter_tuning_string + """<html><p>&emsp;&emsp;&emsp;&emsp;BatchNormalization</p><html>"""
            count2 = 1
        if keyword23 in data2:
            Hyperparameter_tuning_string = Hyperparameter_tuning_string + """<html><p>&emsp;&emsp;&emsp;&emsp;Early Stopping</p><html>"""
            count3 = 1
           
        if keyword22 in data2:
            Hyperparameter_tuning_string += """<html><p>&emsp;&emsp;&emsp;&emsp;Data Augmentation: ImageDataGenerator</p><html>"""
            count4 = 1
           
        if count1 == 0 or count2 == 0 or count3 == 0 or count4 == 0:
            Hyperparameter_tuning_string1 = """<html><p>&emsp;&emsp;&emsp;&emsp;Try using techniques like: </p><html>"""
            if count1 == 0:
                Hyperparameter_tuning_string1 += """<html><p style="color:#008000";>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Increasing Dropout Rate </p><html>"""
            if count2 == 0:
                Hyperparameter_tuning_string1 += """<html><p style="color:#008000";>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;BatchNormalization </p><html>"""
            if count3 == 0:
                Hyperparameter_tuning_string1 += """<html><p style="color:#008000";>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Early Stopping </p><html>"""
            if count4 == 0:
                Hyperparameter_tuning_string1 += """<html><p style="color:#008000";>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;ImageDataGenerator(Data Augmentation) </p><html>"""
               
       # print(Hyperparameter_tuning_string)
               
        if count1 == 0 or count2 == 0 or count3 == 0 or count4 == 0:
            #str1 =  "\n\tParameters used while tuning the model to improve the accuracy score:"
            str1 = """<html><p>&emsp;&emsp;&emsp;&emsp;Dropout - increasing rate </p><p>&emsp;&emsp;&emsp;&emsp;Batch Normalization </p><p>&emsp;&emsp;&emsp;&emsp;Early Stopping - (patience = """ + str(patience_value) + ", monitor = " + monitor_value +")" + "</p><p>&emsp;&emsp;&emsp;&emsp;ImageDataGenerator(Data Augmentation)</p></html>"
   
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
g = open('AI_Glassbox_Image_Classification_Model_Review1.html', 'w')
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
print("""<html><p><h2>AI Glassbox - Image Classification Model Review Report</h2></p></html>""")
print("""<html><br></html>""")
print("""<html><p><span><b>CheckPoint 1:</span></b><span> Data Exploration:</span></p></html>""")
labels = parser.get('variables1', 'labels')
num_classes = parser.get('variables1', 'num_in each_class')
labels = 'labels = ' + codefilename + "." + labels
num_class = parser.get('variables1', 'num_class')
exec(labels)
num_classes = 'num_classes = ' + codefilename + "." + num_classes
exec(num_classes)
num_class = 'num_class = ' + codefilename + "." + num_class
exec(num_class)
print("""<html><p>&emsp;&emsp;Labels present in data : """ + str(labels) + "</p></html>""")
print("""<html><p>&emsp;&emsp;Number of Images contained in each Label : """ + str(num_classes) + "</p></html>""")
print("""<html><p>&emsp;&emsp;No eda script found in the given code</p></html>""")
###############################################
print("""<html><p><span><b>CheckPoint 2:</span></b><span> Feature Engineering:</span></p></html>""")

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

encoder_matches_string = """<html><p>&emsp;&emsp;Encoder techniques check:</p></html>"""
if len(encoder_matches) > 0:
    for i in encoder_matches:
        encoder_matches_string += """<html><p>&emsp;&emsp;&emsp;&emsp;""" + str(i) + "</p></html>"""
else:
    count = 0
    if keyword1 in data2:
        encoder_matches_string = encoder_matches_string + """<html><p>&emsp;&emsp;&emsp;&emsp;LabelEncoder</p></html>"""
        count += 1
    if keyword2 in data2:
        encoder_matches_string = encoder_matches_string + """<html><p>&emsp;&emsp;&emsp;&emsp;OneHotEncoder</p></html>"""
        count += 1
    if count == 0:
        encoder_matches_string += """<html><span style="color:#FF0000";>;&emsp;&emsp;&emsp;&emsp;No Encoder techniques applied. </span> <span style="color:#008000";>Try using techniques like: LabelEncoder,OneHotEncoder</p></html>"""
 
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
model_matches_string = """<html><p>&emsp;&emsp;Model used check:</p></html>"""
if len(model_matches) > 0:
    for i in model_matches:
        model_matches_string +=  """<html><p>&emsp;&emsp;&emsp;&emsp;""" + str(i) + "</p></html>"""
else:
    count = 0
    if keyword6 in data2:
        model_matches_string = model_matches_string + """<html><p>&emsp;&emsp;&emsp;&emsp;Sequential</p></html>"""
        count += 1
   
    if count == 0:
        model_matches_string = model_matches_string + """<html><span style="color:#FF0000";>&emsp;&emsp;&emsp;&emsp;No Model api have applied. </span> <span style="color:#008000";>Try using Sequential,Functional API from keras</span></html>"""

       
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


holdouts_string = """<html><p>&emsp;&emsp;Data Splitting techniques check:</p></html>"""
if len(holdouts1) > 0:
    for i in holdouts1:
        holdouts_string +=  """<html><p>&emsp;&emsp;&emsp;&emsp;""" + str(i) + "</p></html>"""
else:
    if keyword3 in data2:
        holdouts_string = holdouts_string + """<html><p>&emsp;&emsp;&emsp;&emsp;Kfold</p></html>"""
    if keyword4 in data2:
        holdouts_string = holdouts_string + """<html><p>&emsp;&emsp;&emsp;&emsp;Bootstrap</p></html>"""

if len(holdouts2) > 0:
    for i in holdouts2:
        holdouts_string +=  """<html><p>&emsp;&emsp;&emsp;&emsp;""" + str(i) + "</p></html>"""
else:
    if keyword5 in data2:
        holdouts_string = holdouts_string + """<html><p>&emsp;&emsp;&emsp;&emsp;train_test_split</p></html>"""

if (len(holdouts1) == 0) and (len(holdouts2) == 0):
    if (keyword3 not in data2) and (keyword4 not in data2) and (keyword5 not in data2):
        holdouts_string = holdouts_string + """<html><span style="color:#FF0000";>&emsp;&emsp;&emsp;&emsp;No Data splitting techniques were applied. </span> <span style="color:#008000";>Try using techniques like: Train Test Split, Kfold Cross Validation, Bootstrap</span></html>"""

       
print(holdouts_string)
############################################################
result_layer_df = pd.DataFrame(data=None,columns=['Layer', 'used'])

print("""<html><span><b>CheckPoint 3:</span></b><span> CNN and Metrics:</span></html>""")
print("""<html><br></html>""")
print("""<html><br></html>""")
print("""<html>&emsp;&emsp;Different Layers Used:</html>""" )
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
       model_layers_matches_string +=  """<html><p>&emsp;&emsp;""" + str(i) + "</p></html>"""
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
        model_layers_matches_string = model_layers_matches_string + """<html><span style="color:#FF0000";>&emsp;&emsp;No Convolution layers used. </span> <span style="color:#008000";>&emsp;Try using layers like: Conv2D, Conv2DTranspose </span></html>"""

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
       model_layers_matches_string1 +=  """<html><p>&emsp;&emsp;""" + str(i) + "</p></html>"""
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
        model_layers_matches_string1 = model_layers_matches_string1 + 'No core layers used. Try using layers like: Dense, Embedding '


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
       model_layers_matches_string2 += """<html><p>&emsp;&emsp;""" + str(i) + "</p></html>"""
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
        model_layers_matches_string2 = model_layers_matches_string2 + 'No pooling layers used. Try using layers like: AveragePooling2D, MaxPooling2D '

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
       model_layers_matches_string3 += """<html><p>&emsp;&emsp;""" + str(i) + "</p></html>"""
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
        model_layers_matches_string3 = model_layers_matches_string3 + 'No Regularization layers used. Try using layers like: Dropout, GaussianDropout '

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
       model_layers_matches_string4 += """<html><p>&emsp;&emsp;""" + str(i) + "</p></html>"""
      #print(model_layers_matches_string)
else:
    count = 0
    if keyword21 in data2:
        model_layers_matches_string4 = model_layers_matches_string4 + 'BatchNormalization'
        count += 1
    if keyword31 in data2:
        model_layers_matches_string4 = model_layers_matches_string4 + 'LayerNormalization'
        count += 1
    if count == 0:
        model_layers_matches_string4 = model_layers_matches_string4 + ' No Normalization layers used. Try using layers like: BatchNormalization, LayerNormalization '

df_layer = pd.DataFrame({'Layer': ['Normalization Layers'],
                         'used': [model_layers_matches_string4]})
result_layer_df  = result_layer_df.append(df_layer, ignore_index=True)
#print(result_layer_df)
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
       model_layers_matches_string5 += """<html><p>&emsp;&emsp;""" + str(i) + "</p></html>"""
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
        model_layers_matches_string5 = model_layers_matches_string5 + 'No Reshaping layers used. Try using layers like:Reshape, Flatten '

            #print(model_layers_matches_string5)
df_layer = pd.DataFrame({'Layer': ['Reshaping Layers'],
                         'used': [model_layers_matches_string5]})
result_layer_df  = result_layer_df.append(df_layer, ignore_index=True)
pdtabulate=lambda df:tabulate(result_layer_df,headers='keys',tablefmt='psql')
#html = result_layer_df.to_html()
#print("""<html>""" + html + "</html>""")
with open('report2.txt','w') as fh3:
    fh3.write(pdtabulate(result_layer_df))
print("""<html><iframe src="report2.txt" frameborder="0" height="170" width="95%"></iframe></html>""")

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

activation_matches_string = """<html><p>&emsp;&emsp;Activation techniques applied:</p></html>"""
if len(activation_matches) > 0:
    for i in activation_matches:
       activation_matches_string += """<html><p>&emsp;&emsp;&emsp;&emsp;""" + str(i) + "</p></html>"""
       
else:
    count = 0
    if keyword11 in data2:
       
        activation_matches_string = activation_matches_string + """<html><p>&emsp;&emsp;&emsp;&emsp;relu</p></html>"""
        count += 1
    if keyword12 in data2:
        activation_matches_string = activation_matches_string + """<html><p>&emsp;&emsp;&emsp;&emsp;sigmoid</p></html>"""
        count += 1
    if keyword13 in data2:
        activation_matches_string = activation_matches_string + """<html><p>&emsp;&emsp;&emsp;&emsp;softmax</p></html>"""
        count += 1
    if keyword14 in data2:
        activation_matches_string = activation_matches_string + """<html><p>&emsp;&emsp;&emsp;&emsp;Tanh</p></html>"""
        count += 1
    if keyword15 in data2:
        activation_matches_string = activation_matches_string + """<html><p>&emsp;&emsp;&emsp;&emsp;softsign</p></html>"""
        count += 1
    if count == 0:
        activation_matches_string = activation_matches_string + """<html><span style="color:#FF0000";>&emsp;&emsp;&emsp;&emsp;No Activation functions applied. </span> <span style="color:#008000";>Try using techniques like: relu, softmax </span></html>"""

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

img_aug_matches_string = """<html><p>&emsp;&emsp;Data Augumentation Used:</p></html>"""
if len(img_aug_matches) > 0:
    for i in img_aug_matches:
       img_aug_matches_string += """<html><p>&emsp;&emsp;&emsp;&emsp;""" + str(i) + "</p></html>"""
      # print(model_layers_matches_string)
else:
    count = 0
    if keyword22 in data2:
        img_aug_matches__string = img_aug_matches_string + """<html><p>&emsp;&emsp;&emsp;&emsp;ImageDataGenerator</p></html>"""
        count += 1
    if count == 0:
        img_aug_matches_string = img_aug_matches_string +  """<html><span style="color:#FF0000";>&emsp;&emsp;&emsp;&emsp;No Augmentation techniques used. </span> <span style="color:#008000";>Try using techniques like: ImageDataGenerator </span></html>"""

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

CNN_metric_matches_string = """<html><p>&emsp;&emsp;CNN Metrics applied:</p></html>"""
if len(CNN_metric_matches) > 0:
    for i in CNN_metric_matches:
        CNN_metric_matches_string +="""<html><p>&emsp;&emsp;&emsp;&emsp;""" + str(i) + "</p></html>"""
       
else:
    count = 0
    if keyword7 in data2:
        CNN_metric_matches_string = CNN_metric_matches_string + """<html><p>&emsp;&emsp;&emsp;&emsp;accuracy_score</p></html>"""
        count += 1
    if keyword8 in data2:
        CNN_metric_matches_string = CNN_metric_matches_string + """<html><p>&emsp;&emsp;&emsp;&emsp;precision_score</p></html>"""
        count += 1
    if keyword9 in data2:
        CNN_metric_matches_string = CNN_metric_matches_string + """<html><p>&emsp;&emsp;&emsp;&emsp;recall_score</p></html>"""
        count += 1
    if keyword10 in data2:
        CNN_metric_matches_string = CNN_metric_matches_string + """<html><p>&emsp;&emsp;&emsp;&emsp;classification_report</p></html>"""
        count += 1
    if keyword24 in data2:
        CNN_metric_matches_string = CNN_metric_matches_string + """<html><p>&emsp;&emsp;&emsp;&emsp;f1_score</p></html>"""
        count += 1
    if count == 0:
        CNN_metric_matches_string = CNN_metric_matches_string + """<html><span style="color:#FF0000";>&emsp;&emsp;&emsp;&emsp;No Metrics have applied. </span> <span style="color:#008000";>Try using accuracy_score,f1_score,precison_score </span></html>"""
print(CNN_metric_matches_string)

optimizer_var = defaultparser.get('model_param','optimizer')
loss_var = defaultparser.get('model_param','loss')
epochs_var = defaultparser.getint('model_param','epochs')
batchsize_var = defaultparser.getint('model_param','batch_size')
print("""<html><p>&emsp;&emsp;Model Parameters Used:</p></html>""")
print("""<html><p>&emsp;&emsp;&emsp;&emsp;Loss function : """ + loss_var + "</p></html>""")
print("""<html><p>&emsp;&emsp;&emsp;&emsp;Optimizer : """ + optimizer_var + "</p></html>""")
print("""<html><p>&emsp;&emsp;&emsp;&emsp;Epochs : """ + str(epochs_var) + "</p></html>""")
print("""<html><p>&emsp;&emsp;&emsp;&emsp;Batch size : """ + str(batchsize_var) + "</p></html>""")


########################################################################

result_df = pd.DataFrame(data=None,
                         columns=['Model', 'OD_acc_test','OD_acc_val','OD_f1_test','OD_f1_val',
                                  'Train Accuracy', 'Test Accuracy', 'val Accuracy','Train F1', 'Test F1','val F1',
                                  'Train Precision', 'Test Precision', 'val Precision','Train Recall', 'Test Recall','val Recall'])
baseline = pd.DataFrame(data = None,
                          columns = ['Metrics','Accuracy','F1-Score','Precision','Recall'])
     
       

if 'Sequential' in model_matches:
    modelused = data[data['Type'] == 'Sequential']['Variable']

    modelused = pd.DataFrame(data=modelused)
    modelused = modelused.iloc[0]
    modelused = str(modelused[0])
    print("""<html><p>&emsp;&emsp;Baseline Model Summary:</p></html>""")
    with open('report.txt','w') as fh:
        model_summary = "model_summary = " + codefilename + "." + modelused + ".summary(print_fn=lambda x: fh.write(x + '\\n'))"
        exec(model_summary)
    print("""<html><p><iframe src="report.txt" frameborder="0" height="400" width="95%"></iframe></p></html>""")      
   
    predictions = "predictions = " + codefilename + "." + modelused
       
    exec(predictions)
   
    y_train_pred = predictions.predict_classes(X_train)
    y_train_actual = np.argmax(y_train, axis = 1)
   
    train_acc_baseline = accuracy_score(y_train_pred,y_train_actual)
    train_f1_baseline = f1_score(y_train_actual, y_train_pred,average='weighted')
    train_precision_baseline = precision_score(y_train_actual, y_train_pred,average='weighted')
    train_recall_baseline = recall_score(y_train_actual, y_train_pred,average='weighted')
   
   
    train_acc_baseline_g = float(str.format('{0:.2f}',  train_acc_baseline))
    train_f1_baseline_g = float(str.format('{0:.2f}',  train_f1_baseline))
    train_precision_baseline_g = float(str.format('{0:.2f}',  train_precision_baseline))
    train_recall_baseline_g = float(str.format('{0:.2f}',  train_recall_baseline))
   
    print("""<html><p>&emsp;&emsp;Metrics for Baseline model:</p></html>""")
     
    y_test_pred = predictions.predict_classes(X_test)
    y_test_actual = np.argmax(y_test, axis = 1)
   
    test_acc_baseline = accuracy_score(y_test_pred, y_test_actual)
    test_f1_baseline = f1_score(y_test_actual, y_test_pred,average='weighted')
    test_precision_baseline = precision_score(y_test_actual, y_test_pred,average='weighted')
    test_recall_baseline = recall_score(y_test_actual, y_test_pred,average='weighted')
   
    test_acc_baseline_g = float(str.format('{0:.2f}',  test_acc_baseline))
    test_f1_baseline_g = float(str.format('{0:.2f}',  test_f1_baseline))
    test_precision_baseline_g = float(str.format('{0:.2f}',  test_precision_baseline))
    test_recall_baseline_g = float(str.format('{0:.2f}',  test_recall_baseline))
   
    

    df_baseline = pd.DataFrame({'Model': ['baseline model'],
                                'Train Accuracy': [train_acc_baseline], 'Test Accuracy': [test_acc_baseline],
                                'Train F1': [train_f1_baseline], 'Test F1': [test_f1_baseline],
                                'Train Precision': [train_precision_baseline], 'Test Precision': [test_precision_baseline],
                                'Train Recall': [train_recall_baseline],'Test Recall' :[test_recall_baseline]})
   
    train = pd.DataFrame({'Metrics' : ['train'],
                          'Accuracy':[train_acc_baseline],
                          'F1-Score':[train_f1_baseline],
                          'Precision':[train_precision_baseline],
                          'Recall':[train_recall_baseline]})
    baseline = baseline.append(train,ignore_index=True)
   
    test = pd.DataFrame({'Metrics' : ['test'],
                          'Accuracy':[test_acc_baseline],
                          'F1-Score':[test_f1_baseline],
                          'Precision':[test_precision_baseline],
                          'Recall':[test_recall_baseline]})
    baseline = baseline.append(test,ignore_index=True)
   
    baseline = baseline.set_index(['Metrics'])
    baseline = baseline.T
   
    result_html2 = baseline.to_html()
    print("""<html>""" + result_html2 + "</html>""")
   
    result_df = result_df.append(df_baseline, ignore_index=True)
    #print(result_df)

   
################################################################
# HYPERPARAMETR_Check = True

hyperparameter_Check = parser.get('hyperparameter_checks', 'hyperparameter_check')
overfitting_check = float(parser.get('overfitting_checks', 'overfitting_value'))

if hyperparameter_Check == 'True':
    print("""<html><p><b><span>CheckPoint 4:</span></b><span> Hyperparameter Tuning:</span></b></p></html>""")
   # """<html><p>&emsp;&emsp;Model used check:</p></html>"""
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
   
             
   # print(tunedmodel.summary())
    # load the saved model
    saved_model = load_model('tuned_model.h5')
   
    y_train_pred = saved_model.predict_classes(X_train)
    y_train_actual = np.argmax(y_train, axis = 1)
    train_acc_best = accuracy_score(y_train_pred,y_train_actual)
    train_f1_best = f1_score(y_train_actual, y_train_pred,average='weighted')
    train_precision_best = precision_score(y_train_actual, y_train_pred,average='weighted')
    train_recall_best = recall_score(y_train_actual, y_train_pred,average='weighted')
       
    train_acc_best_g = float(str.format('{0:.2f}',  train_acc_best))
    train_f1_best_g = float(str.format('{0:.2f}',  train_f1_best))
    train_precision_best_g = float(str.format('{0:.2f}',  train_precision_best))
    train_recall_best_g = float(str.format('{0:.2f}',  train_recall_best))
    #print('the accuracy obtained on the best train set is:', train_acc_best)
    #print(classification_report(y_train_actual, y_train_pred))
   
    y_test_pred = saved_model.predict_classes(X_test)
    y_test_actual = np.argmax(y_test, axis = 1)
    test_acc_best = accuracy_score(y_test_pred,y_test_actual)
    test_f1_best = f1_score(y_test_actual, y_test_pred,average='weighted')
    test_precision_best = precision_score(y_test_actual, y_test_pred,average='weighted')
    test_recall_best = recall_score(y_test_actual, y_test_pred,average='weighted')
   
    test_acc_best_g = float(str.format('{0:.2f}',  test_acc_best))
    test_f1_best_g = float(str.format('{0:.2f}',  test_f1_best))
    test_precision_best_g = float(str.format('{0:.2f}',  test_precision_best))
    test_recall_best_g = float(str.format('{0:.2f}',  test_recall_best))
   
   
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
   
    tuned = pd.DataFrame(data = None,
                          columns = ['Metrics','Accuracy','F1-Score','Precision','Recall'])
     
    if float(train_acc_baseline - test_acc_baseline) > overfitting_check:
        print("""<html><b><p style="color:#FF0000";>&emsp;&emsp;The baseline model's performance metrics values show the model is tending towards overfitting</p></b></html>""")
        if (train_acc_best - test_acc_best) < overfitting_check:
            print("""<html><p style="color:#008000";>&emsp;&emsp;Consider tuning the model hyper parameters. Below parameters can be used to tune hyper parameters in the base line model to avoid over fitting:</p></html>""")
            print(param3)
            #print('\t\t', end='')
            print("""<html><p>&emsp;&emsp;Tuned Model Summary:</p></html>""")
            with open('report1.txt','w') as fh1:
                tuned_model_summary = tunedmodel.summary(print_fn=lambda x: fh1.write(x + '\n'))
            print("""<html>
                  <iframe src="report1.txt" frameborder="0" height="400" width="95%" ></iframe>
                  </html>""")
           
            print("""<html><p>&emsp;&emsp;Metrics for tuned model:</p></html>""")
            train = pd.DataFrame({'Metrics' : ['train'],
                          'Accuracy':[train_acc_best],
                          'F1-Score':[train_f1_best],
                          'Precision':[train_precision_best],
                          'Recall':[train_recall_best]})
            tuned = tuned.append(train,ignore_index=True)
   
            test = pd.DataFrame({'Metrics' : ['test'],
                          'Accuracy':[test_acc_best],
                          'F1-Score':[test_f1_best],
                          'Precision':[test_precision_best],
                          'Recall':[test_recall_best]})
            tuned = tuned.append(test,ignore_index=True)
   
            tuned = tuned.set_index(['Metrics'])
            tuned = tuned.T
   
            result_html2 = tuned.to_html()
            print("""<html>""" + result_html2 + "</html>""")
       
       
        else:
            print("""<html><b><p style="color:#FF0000";>&emsp;&emsp;Please tweak the hyper parameters and re-train model</p></b></html>""")
           

    elif test_acc_best > test_acc_baseline:
        print("""<html><p style="color:#008000";>&emsp;&emsp;Consider tuning the model hyper parameters. Below parameters can be used to tune hyper parameters in the base line model to improve the accuracy score:</p></html>""")
        #print('\t\t', end='')
        print(param3)
        print("""<html><p>&emsp;&emsp;Tuned Model Summary:</p></html>""")
        with open('report1.txt','w') as fh1:
            tuned_model_summary = tunedmodel.summary(print_fn=lambda x: fh1.write(x + '\n'))
        print("""<html>
              <iframe src="report1.txt" frameborder="0" height="400" width="95%" ></iframe>
              </html>""")
        
        print("""<html><p>&emsp;&emsp;Metrics for tuned model:</p></html>""")

        train = pd.DataFrame({'Metrics' : ['train'],
                          'Accuracy':[train_acc_best],
                          'F1-Score':[train_f1_best],
                          'Precision':[train_precision_best],
                          'Recall':[train_recall_best]})
        tuned = tuned.append(train,ignore_index=True)
   
        test = pd.DataFrame({'Metrics' : ['test'],
                          'Accuracy':[test_acc_best],
                          'F1-Score':[test_f1_best],
                          'Precision':[test_precision_best],
                          'Recall':[test_recall_best]})
        tuned = tuned.append(test,ignore_index=True)
   
        tuned = tuned.set_index(['Metrics'])
        tuned = tuned.T
   
        result_html2 = tuned.to_html()
        print("""<html>""" + result_html2 + "</html>""")
       
    else:
        print("""<html><b><p style="color:#008000";>&emsp;&emsp;The baseline model has been trained using tuned parameters</p></b></html>""")
    ########
   
    print("""<html><br></html>""")
    print("""<html><p><b>&emsp;&emsp;Various metrics for val data using baseline and tuned model</b></p></html>""")
   
     
   
    y_val_pred = predictions.predict_classes(X_val)
    y_val_actual = np.argmax(y_val, axis = 1)
   
     
   
    val_acc_baseline = accuracy_score(y_val_pred,y_val_actual)
    val_f1_baseline = f1_score(y_val_actual, y_val_pred,average='weighted')
    val_precision_baseline = precision_score(y_val_actual, y_val_pred,average='weighted')
    val_recall_baseline = recall_score(y_val_actual, y_val_pred,average='weighted')
   
    val_acc_baseline_g = float(str.format('{0:.2f}',  val_acc_baseline))
    val_f1_baseline_g = float(str.format('{0:.2f}',  val_f1_baseline))
    val_precision_baseline_g = float(str.format('{0:.2f}',  val_precision_baseline))
    val_recall_baseline_g = float(str.format('{0:.2f}',  val_recall_baseline))
   
   
    result_df.loc[result_df['Model'] == 'baseline model', ['val Accuracy', 'val F1', 'val Precision','val Recall']] = val_acc_baseline, val_f1_baseline, val_precision_baseline, val_recall_baseline
                   
    y_val_pred = saved_model.predict_classes(X_val)
    y_val_actual = np.argmax(y_val, axis = 1)
    val_acc_best = accuracy_score(y_val_pred,y_val_actual)
    val_f1_best = f1_score(y_val_actual, y_val_pred,average='weighted')
    val_precision_best = precision_score(y_val_actual, y_val_pred,average='weighted')
    val_recall_best = recall_score(y_val_actual, y_val_pred,average='weighted')
   
    val_acc_best_g = float(str.format('{0:.2f}',  val_acc_best))
    val_f1_best_g = float(str.format('{0:.2f}',  val_f1_best))
    val_precision_best_g = float(str.format('{0:.2f}',  val_precision_best))
    val_recall_best_g = float(str.format('{0:.2f}',  val_recall_best))
       
    
    val = pd.DataFrame(data = None,
                         columns = ['Model','Accuracy','F1-Score','Precision','Recall'])
     
   
    base = pd.DataFrame({'Model' : ['baseline'],
                          'Accuracy':[val_acc_baseline],
                          'F1-Score':[val_f1_baseline],
                          'Precision':[val_precision_baseline],
                          'Recall':[val_recall_baseline]})
    val = val.append(base,ignore_index=True)
   
    best = pd.DataFrame({'Model' : ['best'],
                          'Accuracy':[val_acc_best],
                          'F1-Score':[val_f1_best],
                          'Precision':[val_precision_best],
                          'Recall':[val_recall_best]})
    val = val.append(best,ignore_index=True)
   
    val = val.set_index(['Model'])
    val = val.T
   
    result_html2 = val.to_html()
    print("""<html>""" + result_html2 + "</html>""")
       
    result_df.loc[result_df['Model'] == 'tuned model', ['val Accuracy', 'val F1', 'val Precision','val Recall']] = val_acc_best, val_f1_best, val_precision_best, val_recall_best

   
    print("""<html><br></html>""")
    print("""<html><b><p>&emsp;&emsp;Overfitting degree for baseline and tuned model</p></b></html>""")
   
    print("""<html><p>&emsp;&emsp;Baseline model Accuracy:</p></html>""")
   
    abs_od_acc_test_baseline = train_acc_baseline - test_acc_baseline
    per_od_acc_test_baseline = ((test_acc_baseline - train_acc_baseline) / train_acc_baseline) * 100            
    if abs_od_acc_test_baseline > overfitting_check:
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From train to test: </span><b><span style="color:#FF0000";>""" + str.format('{0:.3f}', abs_od_acc_test_baseline) + "</span></b></p></html>""")
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From train to test(perc change):  </span><b><span style="color:#FF0000";>""" + str.format('{0:.3f}', per_od_acc_test_baseline) + "</span></b></p></html>""")
    elif((abs_od_acc_test_baseline < overfitting_check) and (abs_od_acc_test_baseline > 0)):
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From train to test: </span><b><span style="color:#008000";>""" + str.format('{0:.3f}', abs_od_acc_test_baseline) + "</span></b></p></html>""")
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From train to test(perc change): </span><b><span>""" + str.format('{0:.3f}', per_od_acc_test_baseline) + "</span></b></p></html>""")
    else:
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From train to test: </span><b><span>""" + str.format('{0:.3f}', abs_od_acc_test_baseline) + "</span></b></p></html>""")
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From train to test(perc change): </span><b><span>""" + str.format('{0:.3f}', per_od_acc_test_baseline) + "</span></b></p></html>""")

    abs_od_acc_val_baseline = test_acc_baseline - val_acc_baseline
    per_od_acc_val_baseline = ((val_acc_baseline - test_acc_baseline) / test_acc_baseline) * 100
    #print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From test to val: </span><b><span style="color:#008000";>""" + str.format('{0:.3f}', abs_od_acc_val_baseline) + "</span></b></p></html>""")
    #print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From test to val(perc change): </span><b><span style="color:#008000";>""" + str.format('{0:.3f}', per_od_acc_val_baseline) + "</span></b></p></html>""")
    if abs_od_acc_val_baseline > overfitting_check:
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From test to val: </span><b><span style="color:#FF0000";>""" + str.format('{0:.3f}', abs_od_acc_val_baseline) + "</span></b></p></html>""")
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From test to val(perc change):  </span><b><span style="color:#FF0000";>""" + str.format('{0:.3f}', per_od_acc_val_baseline) + "</span></b></p></html>""")
    elif((abs_od_acc_val_baseline < overfitting_check) and (abs_od_acc_val_baseline > 0)):
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From test to val: </span><b><span style="color:#008000";>""" + str.format('{0:.3f}', abs_od_acc_val_baseline) + "</span></b></p></html>""")
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From test to val(perc change): </span><b><span>""" + str.format('{0:.3f}', per_od_acc_val_baseline) + "</span></b></p></html>""")
        
    else:
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From test to val: </span><b><span>""" + str.format('{0:.3f}', abs_od_acc_val_baseline) + "</span></b></p></html>""")
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From test to val(perc change): </span><b><span>""" + str.format('{0:.3f}', per_od_acc_val_baseline) + "</span></b></p></html>""")
    
    print("""<html><p>&emsp;&emsp;Baseline model F1 Score</p></html>""")
   
    abs_od_f1_test_baseline = train_f1_baseline - test_f1_baseline
    per_od_f1_test_baseline = ((test_f1_baseline - train_f1_baseline) / train_f1_baseline) * 100
    if abs_od_f1_test_baseline > overfitting_check:
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From train to test: </span><b><span style="color:#FF0000";>""" + str.format('{0:.3f}', abs_od_f1_test_baseline) + "</span></b></p></html>""")
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From train to test(perc change):  </span><b><span style="color:#FF0000";>""" + str.format('{0:.3f}', per_od_f1_test_baseline) + "</span></b></p></html>""")
    elif((abs_od_f1_test_baseline < overfitting_check) and (abs_od_f1_test_baseline > 0)):
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From train to test: </span><b><span style="color:#008000";>""" + str.format('{0:.3f}', abs_od_f1_test_baseline) + "</span></b></p></html>""")
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From train to test(perc change): </span><b><span>""" + str.format('{0:.3f}', per_od_f1_test_baseline) + "</span></b></p></html>""")

    else:
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From train to test: </span><b><span>""" + str.format('{0:.3f}', abs_od_f1_test_baseline) + "</span></b></p></html>""")
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From train to test(perc change): </span><b><span>""" + str.format('{0:.3f}', per_od_f1_test_baseline) + "</span></b></p></html>""")
    abs_od_f1_val_baseline = test_f1_baseline - val_f1_baseline
    per_od_f1_val_baseline = ((val_f1_baseline - test_f1_baseline) / test_f1_baseline) * 100
    if abs_od_f1_val_baseline > overfitting_check:
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From test to val: </span><b><span style="color:#FF0000";>""" + str.format('{0:.3f}', abs_od_f1_val_baseline) + "</span></b></p></html>""")
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From test to val(perc change):  </span><b><span style="color:#FF0000";>""" + str.format('{0:.3f}', per_od_f1_val_baseline) + "</span></b></p></html>""")
    elif((abs_od_f1_val_baseline < overfitting_check) and (abs_od_f1_val_baseline > 0)):
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From test to val: </span><b><span style="color:#008000";>""" + str.format('{0:.3f}', abs_od_f1_val_baseline) + "</span></b></p></html>""")
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From test to val(perc change): </span><b><span>""" + str.format('{0:.3f}', per_od_f1_val_baseline) + "</span></b></p></html>""")
   
    else:
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From test to val: </span><b><span>""" + str.format('{0:.3f}', abs_od_f1_val_baseline) + "</span></b></p></html>""")
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From test to val(perc change): </span><b><span>""" + str.format('{0:.3f}', per_od_f1_val_baseline) + "</span></b></p></html>""")

    result_df.loc[result_df['Model'] == 'baseline model', ['OD_acc_test', 'OD_acc_val', 'OD_f1_test', 'OD_f1_val']] = abs_od_acc_test_baseline, abs_od_acc_val_baseline, abs_od_f1_test_baseline, abs_od_f1_val_baseline
    #print(result_df)
    
    print("""<html><p>&emsp;&emsp;Tuned model Accuracy:</p></html>""")
   
    abs_od_acc_test_best = train_acc_best - test_acc_best
    per_od_acc_test_best = ((test_acc_best - train_acc_best) / train_acc_best) * 100
    if abs_od_acc_test_best > overfitting_check:
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From train to test: </span><b><span style="color:#FF0000";>""" + str.format('{0:.3f}', abs_od_acc_test_best) + "</span></b></p></html>""")
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From train to test(perc change): </span><b><span style="color:#FF0000";>""" + str.format('{0:.3f}', per_od_acc_test_best) + "</span></b></p></html>""")
    elif((abs_od_acc_test_best < overfitting_check) and (abs_od_acc_test_best > 0)):
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From train to test: </span><b><span style="color:#008000";>""" + str.format('{0:.3f}', abs_od_acc_test_best) + "</span></b></p></html>""")
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From train to test(perc change): </span><b><span>""" + str.format('{0:.3f}', per_od_acc_test_best) + "</span></b></p></html>""")
  
    else:
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From train to test: </span><b><span>""" + str.format('{0:.3f}', abs_od_acc_test_best) + "</span></b></p></html>""")
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From train to test(perc change): </span><b><span>""" + str.format('{0:.3f}', per_od_acc_test_best) + "</span></b></p></html>""")

    abs_od_acc_val_best = test_acc_best - val_acc_best
    per_od_acc_val_best = ((val_acc_best - test_acc_best) / test_acc_best) * 100
   
    if abs_od_acc_val_best > overfitting_check:
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From test to val: </span><b><span style="color:#FF0000";>""" + str.format('{0:.3f}', abs_od_acc_val_best) + "</span></b></p></html>""")
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From test to val(perc change): </span><b><span style="color:#FF0000";>""" + str.format('{0:.3f}', per_od_acc_val_best) + "</span></b></p></html>""")
    elif((abs_od_acc_val_best < overfitting_check) and (abs_od_acc_val_best > 0)):
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From test to val: </span><b><span style="color:#008000";>""" + str.format('{0:.3f}', abs_od_acc_val_best) + "</span></b></p></html>""")
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From test to val(perc change): </span><b><span>""" + str.format('{0:.3f}', per_od_acc_val_best) + "</span></b></p></html>""")
    else:
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From test to val: </span><b><span>""" + str.format('{0:.3f}', abs_od_acc_val_best) + "</span></b></p></html>""")
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From test to val(perc change): </span><b><span>""" + str.format('{0:.3f}', per_od_acc_val_best) + "</span></b></p></html>""")
    print("""<html><p>&emsp;&emsp;Tuned model F1 Score:</p></html>""")
    
    abs_od_f1_test_best = train_f1_best - test_f1_best
    per_od_f1_test_best = ((test_f1_best - train_f1_best) / train_f1_best) * 100
    if abs_od_f1_test_best > overfitting_check:
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From train to test: </span><b><span style="color:#FF0000";>""" + str.format('{0:.3f}', abs_od_f1_test_best) + "</span></b></p></html>""")
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From train to test(perc change): </span><b><span style="color:#FF0000";>""" + str.format('{0:.3f}', per_od_f1_test_best) + "</span></b></p></html>""")
    elif((abs_od_f1_test_best < overfitting_check) and (abs_od_f1_test_best > 0)):
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From train to test: </span><b><span style="color:#008000";>""" + str.format('{0:.3f}', abs_od_f1_test_best) + "</span></b></p></html>""")      
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From train to test(perc change): </span><b><span>""" + str.format('{0:.3f}', per_od_f1_test_best) + "</span></b></p></html>""")

    else:
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From train to test: </span><b><span>""" + str.format('{0:.3f}', abs_od_f1_test_best) + "</span></b></p></html>""")      
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From train to test(perc change): </span><b><span>""" + str.format('{0:.3f}', per_od_f1_test_best) + "</span></b></p></html>""")
         
    abs_od_f1_val_best = test_f1_best - val_f1_best
    per_od_f1_val_best = ((val_f1_best - test_f1_best) / test_f1_best) * 100
   
    if abs_od_f1_val_best >  overfitting_check:
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From test to val: </span><b><span style="color:#FF0000";>""" + str.format('{0:.3f}', abs_od_f1_val_best) + "</span></b></p></html>""")
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From test to val(perc change): </span><b><span style="color:#FF0000";>""" + str.format('{0:.3f}', per_od_f1_val_best) + "</span></b></p></html>""")

    elif((abs_od_f1_val_best < overfitting_check) and (abs_od_f1_val_best > 0)):
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From test to val: </span><b><span style="color:#008000";>""" + str.format('{0:.3f}', abs_od_f1_val_best) + "</span></b></p></html>""")
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From test to val(perc change): </span><b><span>""" + str.format('{0:.3f}', per_od_f1_val_best) + "</span></b></p></html>""")

    else:
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From test to val: </span><b><span>""" + str.format('{0:.3f}', abs_od_f1_val_best) + "</span></b></p></html>""")
        print("""<html><p><span>&emsp;&emsp;&emsp;&emsp;From test to val(perc change): </span><b><span>""" + str.format('{0:.3f}', per_od_f1_val_best) + "</span></b></p></html>""")

    result_df.loc[result_df['Model'] == 'tuned model', ['OD_acc_test', 'OD_acc_val', 'OD_f1_test', 'OD_f1_val']] = abs_od_acc_test_best,abs_od_acc_val_best, abs_od_f1_test_best, abs_od_f1_val_best
    
    print("""<html><p><b>Model Metrics Summary</b></p></html>""")
    train_bars = [train_acc_baseline_g, train_acc_best_g]
    test_bars = [test_acc_baseline_g, test_acc_best_g]
    val_bars = [val_acc_baseline_g, val_acc_best_g]
   
    ind2 = np.arange(len(train_bars))
    width2 = 0.1

    fig2, ax2 = plt.subplots(figsize=(7,4))
    r1 = ax2.bar(ind2 - 2*width2, train_bars, width2, label='train')
    r2 = ax2.bar(ind2 - width2, test_bars, width2, label='test')
    r3 = ax2.bar(ind2, val_bars, width2, label='val')
   
    ax2.set_ylabel('Score in Percentage')
    ax2.set_title('Accuracy')
    ax2.set_xticks(ind2)
    ax2.set_xticklabels(('baseline model', 'tuned model'))
    ax2.set_ylim([0,1.1])
    ax2.legend()

    def autolabel2(rects, xpos='center'):
        """
        Attach a text label above each bar in *rects*, displaying its height.

        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """

        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0, 'right': 1, 'left': -1}

        for rect in rects:
            height = rect.get_height()
            ax2.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')

    autolabel2(r1)
    autolabel2(r2)
    autolabel2(r3)
    fig2.tight_layout()
    plt.savefig('metrics2.png')

    print("""<html><img src='metrics2.png'></html>""")
    
    train_bars = [train_f1_baseline_g, train_f1_best_g]
    test_bars = [test_f1_baseline_g, test_f1_best_g]
    val_bars = [val_f1_baseline_g, val_f1_best_g]
    
    ind3 = np.arange(len(train_bars))
    width3 = 0.1

 

    fig3, ax3 = plt.subplots(figsize=(7,4))
    r1 = ax3.bar(ind3 - 2*width2, train_bars, width2, label='train')
    r2 = ax3.bar(ind3 - width2, test_bars, width2, label='test')
    r3 = ax3.bar(ind3, val_bars, width2, label='val')
    
    ax3.set_ylabel('Score in Percentage')
    ax3.set_title('F1 score')
    ax3.set_xticks(ind3)
    ax3.set_xticklabels(('baseline model', 'tuned model'))
    ax3.set_ylim([0,1.1])
    ax3.legend()

 

    def autolabel3(rects, xpos='center'):
        """
        Attach a text label above each bar in *rects*, displaying its height.
        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """
        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0, 'right': 1, 'left': -1}

        for rect in rects:
            height = rect.get_height()
            ax3.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')

    autolabel3(r1)
    autolabel3(r2)
    autolabel3(r3)

    fig3.tight_layout()
    plt.savefig('metrics3.png')
    
    print("""<html><img src='metrics3.png'></html>""")
    ####################
    
    train_bars = [train_precision_baseline_g, train_precision_best_g]
    test_bars = [test_precision_baseline_g, test_precision_best_g]
    val_bars = [val_precision_baseline_g, val_precision_best_g]
    
    ind4 = np.arange(len(train_bars))
    width4 = 0.1
    fig4, ax4 = plt.subplots(figsize=(7,4))
    r1 = ax4.bar(ind4 - 2*width4, train_bars, width4, label='train')
    r2 = ax4.bar(ind4 - width4, test_bars, width4, label='test')
    r3 = ax4.bar(ind4, val_bars, width4, label='val')
    
    ax4.set_ylabel('Score in Percentage')
    ax4.set_title('Precision')
    ax4.set_xticks(ind3)
    ax4.set_xticklabels(('baseline model', 'tuned model'))
    ax4.set_ylim([0,1.1])
    ax4.legend()

 

    def autolabel4(rects, xpos='center'):
        """
        Attach a text label above each bar in *rects*, displaying its height.
        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """
        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0, 'right': 1, 'left': -1}


        for rect in rects:
            height = rect.get_height()
            ax4.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')

    autolabel4(r1)
    autolabel4(r2)
    autolabel4(r3)

    fig4.tight_layout()
    plt.savefig('metrics4.png')

    print("""<html><img src='metrics4.png'></html>""")
    
    train_bars = [train_recall_baseline_g, train_recall_best_g]
    test_bars = [test_recall_baseline_g, test_recall_best_g]
    val_bars = [val_recall_baseline_g, val_recall_best_g]
   
    ind5 = np.arange(len(train_bars))
    width5 = 0.1

    fig5, ax5 = plt.subplots(figsize=(7,4))
    r1 = ax5.bar(ind5 - 2*width2, train_bars, width2, label='train')
    r2 = ax5.bar(ind5 - width2, test_bars, width2, label='test')
    r3 = ax5.bar(ind5, val_bars, width2, label='val')
   
    ax5.set_ylabel('Score in Percentage')
    ax5.set_title('Recall')
    ax5.set_xticks(ind2)
    ax5.set_xticklabels(('baseline model', 'tuned model'))
    ax5.set_ylim([0,1.1])
    ax5.legend()

    def autolabel5(rects, xpos='center'):
        """
        Attach a text label above each bar in *rects*, displaying its height.

        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """

        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0, 'right': 1, 'left': -1}

        for rect in rects:
            height = rect.get_height()
            ax5.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')

    autolabel5(r1)
    autolabel5(r2)
    autolabel5(r3)
    fig5.tight_layout()
    plt.savefig('metrics5.png')

    print("""<html><img src='metrics5.png'></html>""")

#pd.set_option('display.max_columns', None)
print("""<html><p><span><b>CheckPoint 5:</b></span></p></html>""")

retrain_threshold = parser.getint('retrain_metrics','retrain_threshold')
if(abs(((test_f1_baseline - val_f1_baseline)/test_f1_baseline) * 100) > retrain_threshold) :
    print("""<html><p><span>&emsp;&emsp;Considering model's test and validation metrics stats - </span><span style="color:#FF0000";><b>There is a significant difference between the F1 score of Test and Validation dataset. Hence, the model needs to be retrained</b></span></p></html>""")
else:
    print("""<html><p><span>&emsp;&emsp;Considering model's test and validation metrics stats - </span><span style="color:#008000";><b>No retraining required</b></span></p></html""")

print("""<html><p>Model Metrics Summary :</p></html>""")
html1 = result_df.to_html()
print("""<html>""" + html1 + "</html>""")
         
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
    print("AI Glassbox model review Report in AI_Glassbox_Image_Classification_Model_Review1.html file")
    print("AI Glassbox code review Report in AI_Glassbox_Image_Classification_Code_Review.txt file")
    f1.close()

print("Execution time {} seconds ".format(np.round(time.time() - start_time, 2)))


##########################################################