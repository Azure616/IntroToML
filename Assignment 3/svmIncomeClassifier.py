# Starting code for CS6316/4501 HW3, Fall 2016
# By Weilin Xu

import numpy as np
import pandas as pd
import multiprocessing
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import random
import pickle

# Attention: You're not allowed to use the model_selection module in sklearn.
#            You're expected to implement it with your own code.
# from sklearn.model_selection import GridSearchCV

class SvmIncomeClassifier:
    def __init__(self):
        random.seed(0)

    def load_data(self, csv_fpath):
        col_names_x = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                     'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                     'hours-per-week', 'native-country']
        col_names_y = ['label']
        # 1,3,5,11,12,13
        numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                          'hours-per-week']
        categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                            'race', 'sex', 'native-country']
        # 1. Data pre-processing.
        # Hint: Feel free to use some existing libraries for easier data pre-processing.

        df_train = pd.DataFrame(pd.read_csv("./salary.labeled.csv", sep = ', ', engine = 'python'))
        df_train.columns = col_names_x+col_names_y
        # Scaling for numerical columns
        df_train[[col_name for col_name in numerical_cols]] = df_train[[col_name for col_name in numerical_cols]].astype(float)
        for col_name in numerical_cols:
            df_train[col_name] -= df_train[col_name].min()
            df_train[col_name] /= df_train[col_name].max()
        # Turn '?' into 'NaN', easier to handle
        for col_name in categorical_cols:
            for index, feature in enumerate(df_train[col_name]):
                if feature is '?':
                    df_train.set_value(index, col_name, np.NaN)

        # Fill NaN with the closest vaild entry
        df_train = df_train.fillna(method='bfill', axis=0)
        df_train = df_train.fillna(method='ffill', axis=0)
        # Split labels and the rest of the data, encode label
        label_encoder = LabelEncoder()
        train_labels = df_train['label']
        label_encoder.fit(train_labels)
        train_labels = label_encoder.transform(train_labels)
        df_train = df_train.drop('label', axis=1)

        # Also process "salary.2predict.csv" here
        df_pred = pd.DataFrame(pd.read_csv("./salary.2Predict.csv", sep = ', ', engine = 'python'))
        df_pred.columns = col_names_x+col_names_y
        # Scaling for numerical columns
        df_pred[[col_name for col_name in numerical_cols]] = df_pred[[col_name for col_name in numerical_cols]].astype(float)
        for col_name in numerical_cols:
            df_pred[col_name] -= df_pred[col_name].min()
            df_pred[col_name] /= df_pred[col_name].max()
        # Turn '?' into 'NaN', easier to handle
        for col_name in categorical_cols:
            for index, feature in enumerate(df_pred[col_name]):
                if feature is '?':
                    df_pred.set_value(index, col_name, np.NaN)

        # Fill NaN with the closest vaild entry
        df_pred= df_pred.fillna(method='bfill', axis=0)
        df_pred = df_pred.fillna(method='ffill', axis=0)
        # Split labels and the rest of the data, encode label
        pred_encoder = LabelEncoder()
        pred_labels = df_pred['label']
        pred_encoder.fit(pred_labels)
        pred_labels = label_encoder.transform(pred_labels)
        df_pred = df_pred.drop('label', axis=1)

        # Learn a dictionary
        dictionary = {}
        for col_name in categorical_cols:
            cate_col1 = set(df_train[col_name].tolist())
            cate_col2 = set(df_pred[col_name].tolist())
            cate_col = cate_col1.union(cate_col2)
            cate_map = {}
            for index, type in enumerate(cate_col):
                cate_map[type] = index
            dictionary[col_name] = cate_map
        #print(dictionary)

        # Encode categorical features
        oh_encoder = OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13], sparse=False)
        for col_name in categorical_cols:
            for index, element in enumerate(df_train[col_name]):
                val_rep = dictionary[col_name][element]
                df_train.set_value(index, col_name, val_rep)
        for col_name in categorical_cols:
            for index, element in enumerate(df_pred[col_name]):
                val_rep = dictionary[col_name][element]
                df_pred.set_value(index, col_name, val_rep)
        fit_set = np.row_stack([df_train.as_matrix(), df_pred.as_matrix()])
        oh_encoder.fit(fit_set)
        result_train = oh_encoder.transform(df_train.as_matrix())

        result_pred = oh_encoder.transform(df_pred.as_matrix())
        if csv_fpath == "salary.labeled.csv":
            return result_train, train_labels
        else:
            return result_pred, pred_labels

    def train_and_select_model(self, training_csv):
        x_train, y_train = self.load_data(training_csv)
        # 2. Select the best model with cross validation.
        # Grid Search
        # Attention: Write your own hyper-parameter candidates.
        param_set = {'kernel': ['rbf','linear', 'poly'],
                     'C': [2**x for x in np.arange(-2, 2)],
                     'gamma': [2**x for x in np.arange(-2, 2)],
                     'degree': np.arange(1, 2)}
        pool = multiprocessing.Pool()
        num_of_folds = 3
        num_of_instances = int(x_train.shape[0])
        rand_indices = list(range(num_of_instances))
        slice_size = int(num_of_instances/num_of_folds)
        random.shuffle(rand_indices)
        params_and_indices = []
        for kernel in param_set['kernel']:
            print("Training on %s kernel..." % kernel)
            parameters = [[C_p, gamma_p] for C_p in param_set['C'] for gamma_p in param_set['gamma']]
            if kernel == 'poly':
                parameters = [[C_p, gamma_p, deg_p] for C_p, gamma_p in parameters for deg_p in param_set['degree']]
            else:
                parameters = [[C_p, gamma_p, 0] for C_p in param_set['C'] for gamma_p in param_set['gamma']]
            for para1, para2, para3 in parameters:
                # 10 fold cross validation: training
                train_data = []
                test_feature_set = []
                test_label_set = []
                for i in np.arange(0, num_of_folds):
                    # Get training data
                    train_indices = rand_indices[:i*slice_size]+rand_indices[(i+1)*slice_size:]
                    train_features = [x_train[index] for index in train_indices]
                    train_labels = [y_train[index] for index in train_indices]
                    train_data.append([train_features, train_labels, [kernel, para1, para2, para3]])
                    # Get testing data
                    test_indices = rand_indices[i*slice_size: (i+1)*slice_size]
                    test_features = [x_train[index] for index in test_indices]
                    test_labels = [y_train[index] for index in test_indices]
                    test_feature_set.append(test_features)
                    test_label_set.append(test_labels)
                models = pool.map(SvmIncomeClassifier.train_model, train_data)
                # 10 fold cross validation: testing
                sum = 0
                for m_index, model in enumerate(models):
                    score = model.score(test_feature_set[m_index], test_label_set[m_index])
                    sum += score
                avrge_score = sum/num_of_folds
                params_and_indices.append([avrge_score,[kernel, para1, para2, para3]])
                if kernel == 'rbf' or kernel == 'linear':
                    print('Kernel: {} | C: {:4f} | gamma: {:4f} => score: {:4f}'.format(kernel, para1, para2, avrge_score))
                else:
                    print('Kernel: {} | C: {:4f} | gamma: {:4f} | degree: {} => score: {:4f}'.format(kernel, para1, para2, para3, avrge_score))
        # Best model and its score
        params_and_indices.sort(key=lambda x:x[0], reverse=True)
        #results = open('./cv_results.txt', 'wb')
        #pickle.dump(params_and_indices, results)
        cv_score = params_and_indices[0][0]
        params = params_and_indices[0][1]
        model = SVC(kernel=params[0], C=params[1], gamma=params[2], degree=params[3])
        model.fit(x_train, y_train)
        return model, cv_score

    def train_model(data):
        train_features = data[0]
        train_labels = data[1]
        param = data[2]
        kernel = param[0]
        #print(kernel)
        model = None
        if kernel == 'rbf' or kernel == 'linear':
            model = SVC(kernel=kernel, C=param[1], gamma=param[2])
        else:
            model = SVC(kernel=kernel, C=param[1], gamma=param[2], degree=param[3])
        model.fit(train_features, train_labels)
        return model

    def predict(self, test_csv, trained_model):
        x_test, _ = self.load_data(test_csv)
        print(x_test[0].shape)
        predictions = trained_model.predict(x_test)
        print(predictions)
        return predictions

    def output_results(self, predictions):
        # 3. Upload your Python code, the predictions.txt as well as a report to Collab.
        # Hint: Don't archive the files or change the file names for the automated grading.
        with open('predictions.txt', 'w') as f:
            for pred in predictions:
                if pred == 0:
                    f.write('<=50K\n')
                else:
                    f.write('>50K\n')

if __name__ == '__main__':
    training_csv = "salary.labeled.csv"
    testing_csv = "salary.2Predict.csv"
    clf = SvmIncomeClassifier()
    trained_model, cv_score = clf.train_and_select_model(training_csv)
    print("The best model was scored %.2f" % cv_score)
    predictions = clf.predict(testing_csv, trained_model)
    clf.output_results(predictions)


