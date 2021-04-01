# uploaded data manually
# now just checking
#import os
#os.listdir(".")

#import pandas as pd
#df = pd.read_csv('bank-additional-full.csv',sep = ';')
#df.head(2)

# just writing
# Will execute it later
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train-test-split-ratio',type=float, default=0.3)
args, _ = parser.parse_known_args()
print('Received arguments {}'.format(args))
split_ratio = args.train_test_split_ratio

import os
import pandas as pd
input_data_path = os.path.join('/opt/ml/processing/input','bank-additional-full.csv')
df = pd.read_csv(input_data_path,sep =";")

# Then, we remove any line with missing values, as well as duplicate lines:
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

#Then, we count negative and positive samples, and display the class ratio.
#This will tell us how unbalanced the dataset is :

one_class = df[df['y']=='yes']
one_class_count = one_class.shape[0]
zero_class = df[df['y']=='no']
zero_class_count = zero_class.shape[0]
zero_to_one_ratio = zero_class_count/one_class_count
print("Ratio: %.2f" % zero_to_one_ratio)

import numpy as np
df['no_previous_contact'] = np.where(df['pdays'] == 999, 1, 0)

df['not_working'] = np.where(np.in1d(df['job'], ['student', 'retired', 'unemployed']), 1, 0)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('y', axis=1),df['y'],
                                                    test_size=split_ratio, random_state=0)
                                                    
# not working
#from sklearn.compose import make_column_transformer
#from sklearn.preprocessing import StandardScaler,OneHotEncoder

#preprocess = make_column_transformer((['age', 'duration', 'campaign', 'pdays', 'previous'],
#                                      StandardScaler()),(['job', 'marital', 'education', 'default', 'housing',
#                                      'loan','contact', 'month', 'day_of_week','poutcome'],
#                                       OneHotEncoder(sparse=False)))
                                       
#train_features = preprocess.fit_transform(X_train)
#test_features = preprocess.transform(X_test)

# working
# Code link
# https://stackoverflow.com/questions/43798377/one-hot-encode-categorical-variables-and-scale-continuous-ones-simultaneouely

 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder

columns_to_scale = ['age', 'duration', 'campaign', 'pdays', 'previous']
columns_to_encode=['job', 'marital', 'education', 'default', 'housing','loan','contact', 'month', 'day_of_week','poutcome']


# Instantiate encoder/scaler
scaler = StandardScaler()
ohe    = OneHotEncoder(sparse=False)

p = Pipeline(
    [("coltransformer", ColumnTransformer(
        transformers=[
            ("assessments", Pipeline([("scale", scaler)]), columns_to_scale),
            ("ranks", Pipeline([("encode", ohe)]), columns_to_encode),
        ]),
    )]
)

train_features = p.fit_transform(X_train)
test_features = p.transform(X_test)

# making dir
#import os
#train_dir = 'train'
#if not os.path.exists(train_dir):
#    os.makedirs(train_dir)
#
#test_dir = 'test'
#if not os.path.exists(train_dir):
#    os.makedirs(test_dir)

# giving dir location only

train_features_output_path = os.path.join('/opt/ml/processing/train', 'train_features.csv')
##train_features_output_path = os.path.join(train_dir, 'train_features.csv')

train_labels_output_path = os.path.join('/opt/ml/processing/train', 'train_labels.csv')
##train_labels_output_path = os.path.join(test_dir, 'train_labels.csv')

# giving dir location only

test_features_output_path = os.path.join('/opt/ml/processing/test', 'test_features.csv')
##test_features_output_path = os.path.join(train_dir, 'test_features.csv')

test_labels_output_path = os.path.join('/opt/ml/processing/test', 'test_labels.csv')
##test_labels_output_path = os.path.join(test_dir, 'test_labels.csv')

# ??
pd.DataFrame(train_features).to_csv(train_features_output_path, header=False, index=False)
pd.DataFrame(test_features).to_csv(test_features_output_path, header=False, index=False)

# Now saving files to path define above

y_train.to_csv(train_labels_output_path, header=False, index=False)
y_test.to_csv(test_labels_output_path, header=False, index=False)



#bucket = sess.default_bucket()
#prefix = 'sagemaker/DEMO-automl-dm'
#s3_input_data = upload_data(path="./bank-additional/bank-additional-full.csv",
#                            key_prefix=prefix+'input')
                            
#from sagemaker.sklearn.processing import SKLearnProcessor
#sklearn_processor = SKLearnProcessor(framework_version='0.20.0',role=sagemaker.get_execution_role(),
#                                     instance_type='ml.t3.medium',instance_count=1)
                                     
#from sagemaker.processing import ProcessingInput, ProcessingOutput
#sklearn_processor.run(code='preprocessing.py',
#                      inputs=[ProcessingInput(source=input_data,destination='/opt/ml/processing/input')],
#                      outputs=[ProcessingOutput(source='/opt/ml/processing/train',output_name='train_data'),
#                               ProcessingOutput(source='/opt/ml/processing/test',output_name='test_data')],
#                               arguments=['--train-test-split-ratio', '0.2'])
                                                                                                                         

