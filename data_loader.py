import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import OrdinalEncoder


class DataLoader:
    def __init__(self):
        self.pred_func = None
        self.dataset = None
        self.outcome = None
        self.cont_features = None
        self.class_num = None
        self.name = '?'

    def load_data(self):
        return self.pred_func, self.dataset, self.cont_features, self.outcome, self.class_num

    def load_privacy_parameters(self):
        x_np = self.dataset.drop(self.outcome, axis=1).to_numpy()
        y_np = self.dataset[self.outcome].to_numpy()

        data_mins = np.amin(x_np, axis=0)
        data_maxs = np.amax(x_np, axis=0)

        pred_min = np.amin(y_np)
        pred_max = np.amax(y_np)

        data_ints = np.repeat(True, x_np.shape[1])
        for feature_index in range(x_np.shape[1]):
            for value in x_np[:, feature_index]:
                if value % 1 != 0:
                    data_ints[feature_index] = False
                    break

        return data_mins, data_maxs, data_ints, pred_min, pred_max


class HeartDisease(DataLoader):
    def __init__(self):
        super().__init__()
        dirname = os.path.dirname(__file__)
        filename_num = os.path.join(dirname, 'data/framingham.csv')

        all_features = ['sex', 'age', 'education', 'smoker', 'cigs_per_day', 'bp_meds', 'prevalent_stroke',
                        'prevelant_hyp', 'diabetes', 'total_chol', 'sys_bp', 'dia_bp', 'bmi', 'heart_rate',
                        'glucose',
                        'heart_disease_label']

        raise FileNotFoundError('We were unable to upload the data from the Framingham heart study. '
                                'Access can be requested online. The other two data sets are available in this '
                                'repository. Remove "HeartDisease()" from experiments to run them without the heart'
                                'disease data set.')

        data = pd.read_csv(filename_num, names=all_features)

        cont_features = ['age', 'cigs_per_day', 'total_chol', 'sys_bp', 'dia_bp', 'bmi', 'heart_rate', 'glucose']
        outcome = 'heart_disease_label'

        data = data.dropna()
        data = data.drop_duplicates(subset=all_features)

        data = data.astype(float)
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)

        model = RandomForestClassifier(random_state=42)
        model = model.fit(data.drop(outcome, axis=1).to_numpy(), data[outcome].to_numpy())

        self.pred_func = model.predict_proba
        self.dataset = data
        self.outcome = outcome
        self.cont_features = cont_features
        self.class_num = 1
        self.name = 'Heart Disease'


class BikeSharingModel:
    def __init__(self, model):
        self.model = model

    def pred_func(self, x):
        return np.exp(self.model.predict(x))


class BikeSharing(DataLoader):
    def __init__(self):
        # https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
        super().__init__()
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'data/bike_rentals_hourly.csv')

        col_names = ['instant', 'dteday', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
                     'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']

        data = pd.read_csv(filename, names=col_names)
        data = data.drop('instant', axis=1).drop('casual', axis=1).drop('registered', axis=1)

        # drop same columns as ALE paper
        data = data.drop('dteday', axis=1).drop('season', axis=1).drop('temp', axis=1)

        cont_features = ['mnth', 'hr', 'weekday', 'weathersit', 'atemp', 'hum', 'windspeed']
        outcome_name = 'cnt'

        data = data.dropna()
        data = data.astype(float)
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)

        # Neural Network configuration taken from ALE paper
        # Nnet						            sklearn
        #
        # 25 nodes (single hidden layer)	    hidden_layer_sizes
        # linear output activation function	    activation = relu?
        # regularization 0.001 (decay) 	        alpha?

        # random state 1 chosen because resulting ALEs are very close to those in the original ALE paper
        model = Pipeline([('scaler', StandardScaler()), ('nn', MLPRegressor(hidden_layer_sizes=(25,), activation='relu',
                                                                            alpha=0.001, max_iter=400,
                                                                            random_state=1,
                                                                            verbose=False))])
        model = model.fit(data.drop(outcome_name, axis=1).to_numpy(), np.log(data[outcome_name].to_numpy()))

        model = BikeSharingModel(model)

        self.pred_func = model.pred_func
        self.dataset = data
        self.outcome = outcome_name
        self.cont_features = cont_features
        self.class_num = None
        self.name = 'Bike Sharing'


class AdultIncome(DataLoader):
    def __init__(self):
        super().__init__()
        # https://archive.ics.uci.edu/ml/datasets/Adult
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'data/adult.data.csv')

        col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                     'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                     'income']

        data = pd.read_csv(filename, names=col_names)
        data = data.drop('fnlwgt', axis=1)

        # remove missing values
        data = data[data['workclass'] != ' ?']
        data = data[data['occupation'] != ' ?']
        data = data[data['native-country'] != ' ?']

        cont_features = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        outcome_name = 'income'

        # use same category order as UCI website
        data['workclass'] = OrdinalEncoder(categories=[[' Private', ' Self-emp-not-inc', ' Self-emp-inc', ' Federal-gov',
                                ' Local-gov', ' State-gov', ' Without-pay', ' Never-worked']], dtype=int)\
                                .fit_transform(data[['workclass']])

        data['education'] = OrdinalEncoder(categories=[[' Bachelors', ' Some-college', ' 11th', ' HS-grad', ' Prof-school',
                                ' Assoc-acdm', ' Assoc-voc', ' 9th', ' 7th-8th', ' 12th', ' Masters', ' 1st-4th', ' 10th',
                                ' Doctorate', ' 5th-6th', ' Preschool']], dtype=int)\
                                .fit_transform(data[['education']])

        data['marital-status'] = OrdinalEncoder(categories=[[' Married-civ-spouse', ' Divorced', ' Never-married',
                                ' Separated', ' Widowed', ' Married-spouse-absent', ' Married-AF-spouse']], dtype=int)\
                                .fit_transform(data[['marital-status']])

        data['occupation'] = OrdinalEncoder(categories=[[' Tech-support', ' Craft-repair', ' Other-service', ' Sales',
                                ' Exec-managerial', ' Prof-specialty', ' Handlers-cleaners', ' Machine-op-inspct',
                                ' Adm-clerical', ' Farming-fishing', ' Transport-moving', ' Priv-house-serv',
                                ' Protective-serv', ' Armed-Forces']], dtype=int)\
                                .fit_transform(data[['occupation']])

        data['relationship'] = OrdinalEncoder(categories=[[' Wife', ' Own-child', ' Husband', ' Not-in-family',
                                ' Other-relative', ' Unmarried']], dtype=int)\
                                .fit_transform(data[['relationship']])

        data['race'] = OrdinalEncoder(categories=[[' White', ' Asian-Pac-Islander', ' Amer-Indian-Eskimo', ' Other',
                                ' Black']], dtype=int)\
                                .fit_transform(data[['race']])

        data['sex'] = OrdinalEncoder(categories=[[' Female', ' Male']], dtype=int)\
                                .fit_transform(data[['sex']])

        data['native-country'] = OrdinalEncoder(categories=[[' United-States', ' Cambodia', ' England', ' Puerto-Rico',
                                ' Canada', ' Germany', ' Outlying-US(Guam-USVI-etc)', ' India', ' Japan', ' Greece',
                                ' South', ' China', ' Cuba', ' Iran', ' Honduras', ' Philippines', ' Italy', ' Poland',
                                ' Jamaica', ' Vietnam', ' Mexico', ' Portugal', ' Ireland', ' France',
                                ' Dominican-Republic', ' Laos', ' Ecuador', ' Taiwan', ' Haiti', ' Columbia', ' Hungary',
                                ' Guatemala', ' Nicaragua', ' Scotland', ' Thailand', ' Yugoslavia', ' El-Salvador',
                                ' Trinadad&Tobago', ' Peru', ' Hong', ' Holand-Netherlands']], dtype=int)\
                                .fit_transform(data[['native-country']])

        data[outcome_name] = LabelEncoder().fit_transform(data[outcome_name])

        data = data.sample(frac=1, random_state=42).reset_index(drop=True)

        model = RandomForestClassifier(random_state=42)
        model = model.fit(data.drop(outcome_name, axis=1).to_numpy(), data[outcome_name].to_numpy())

        self.pred_func = model.predict_proba
        self.dataset = data
        self.outcome = outcome_name
        self.cont_features = cont_features
        self.class_num = 1
        self.name = 'Adult Income'


if __name__ == '__main__':
    pred_func, data, cont_features, outcome_name, class_num = AdultIncome().load_data()

    print(f'Columns: {data.columns}')

    from private_feature_values import DPQuantiles
    data = data.to_numpy()[:, 0]

    q_ratios = np.array([i * 0.1 for i in range(1,20)])
    DPQuantiles(data, np.amin(data), np.amax(data), 0).test_dp_quantiles(q_ratios, 0.1)