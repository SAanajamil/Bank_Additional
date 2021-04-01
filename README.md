# Bank_Additional
 Processing a dataset with scikit-learn
 Uploading the dataset to Amazon S3
 
 1. Creating a new Jupyter notebook, let's first download and extract the dataset.
 2. Then, we load it with pandas.
 3. Now, let's upload the dataset to Amazon S3. We'll use a default bucket automatically
created by SageMaker in the region we're running in. We'll just add a prefix to keep
things nice and tidy.
Writing a processing script with scikit-learn

1. First, we read our single command-line parameter with the argparse library
(https://docs.python.org/3/library/argparse.html): the ratio for
the training and test datasets. The actual value will be passed to the script by the
SageMaker Processing SDK.

2. We load the input dataset using pandas. At startup, SageMaker Processing
automatically copied it from S3 to a user-defined location inside the container; here,
it is /opt/ml/processing/input.
3. Then, we count negative and positive samples, and display the class ratio. This will
tell us how unbalanced the dataset is.
4. In the job column, we can see three categories (student, retired, and
unemployed) that should probably be grouped to indicate that these customers
don't have a full-time job. Let's add another column.
5. Now, let's split the dataset into training and test sets. Scikit-learn has a convenient
API for this, and we set the split ratio according to a command-line argument
passed to the script.
6. The next step is to scale numerical features and to one-hot encode the categorical
features. We'll use StandardScaler for the former, and OneHotEncoder for
the latter.
7. Then, we process the training and test datasets.
8. Finally, we save the processed datasets, separating the features and labels.
They're saved to user-defined locations in the container, and SageMaker Processing
will automatically copy the files to S3 before terminating the job.
That's it. As you can see, this code is vanilla scikit-learn, so it shouldn't be difficult to adapt
your own scripts for SageMaker Processing. Now let's see how we can actually run this.

