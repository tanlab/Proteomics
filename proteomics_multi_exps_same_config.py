from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import SGD
from keras.models import Model
from tensorflow.keras.models import load_model
from keras import backend as K
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from get_data import GetData

import json
import pandas as pd
import numpy as np
import mlflow.keras
import mlflow
import model_artifact as ma

with open('./L1000CDS_subset.json', 'r') as f:
    L = json.load(f)


def build_model(input_dim, output_dim):
    input_layer = Input(shape=(input_dim,))
    inp = Dropout(0.5)(input_layer)
    _ = Dense(units=256)(inp)
    _ = BatchNormalization()(_)
    _ = Activation('relu')(_)
    bottleneck = Dropout(0.7)(_)

    outputs = []
    for i in range(output_dim):
        _ = Dense(units=32, activation='relu')(bottleneck)
        outputs.append(Dense(units=1, activation='sigmoid')(_))

    model = Model(inputs=input_layer, outputs=outputs)
    opt = SGD(learning_rate=0.001)
    model.compile(optimizer=opt, loss='binary_crossentropy')
    return model

descriptor = 'jtvae'
for helper in ['gcp', 'p100', 'both']:
    for cell_line in ['A549', 'MCF7', 'PC3']:
        for target in ['up', 'dn']:
            mlflow.set_tracking_uri('http://193.140.108.166:5000')
            mlflow.set_experiment(cell_line + '_experiments_min_100_' + helper)
            mlflow.start_run()
            mlflow.set_tag('Author', 'Bahattin')

            if descriptor == 'ecfp':
                mlflow.log_param('useChirality', 'True')
                obj = GetData(L=L, cell_line=cell_line, descriptor=descriptor, n_fold=5, random_state=42,
                              random_genes=False, useChirality=True)
            elif descriptor == 'jtvae':
                mlflow.log_param('useChirality', 'False')
                obj = GetData(L=L, cell_line=cell_line, descriptor=descriptor, n_fold=5, random_state=42,
                              random_genes=False, csv_file='JTVAE_Representations.csv')
            else:
                mlflow.log_param('useChirality', 'False')
                obj = GetData(L=L, cell_line=cell_line, descriptor=descriptor, n_fold=5, random_state=42,
                              random_genes=False)

            mlflow.log_param('Model_Type', 'Multi-Task-NN-v2')
            if target == 'up':
                mlflow.log_param('target', 'up_genes')
                x, y, folds = obj.get_up_genes()
            elif target == 'dn':
                mlflow.log_param('target', 'down_genes')
                x, y, folds = obj.get_down_genes()

            lst_y = []
            for i in range(978):
                lst_y.append(np.count_nonzero(y.iloc[:, i]))
            df = pd.DataFrame({'genes': lst_y})
            indexes = df[df['genes'] >= 100].index.values
            del lst_y, df
            y = y[y.columns[indexes]]
            x.drop(['SMILES'], axis=1, inplace=True)
            folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(x, y.iloc[:, 0]))
            print(y)

            mlflow.log_param('cell_line', cell_line)
            mlflow.log_param('descriptor', descriptor)
            mlflow.log_param('n_fold', '5')
            mlflow.log_param('random_state', '42')
            mlflow.log_param('batch_size', '16')
            mlflow.log_param('epochs', '100')
            mlflow.log_param('optimizer', 'SGD')
            mlflow.log_param('learning_rate', '0.001')
            mlflow.log_param('loss', 'binary_crossentropy')
            mlflow.log_param('early_stopping', 'False')
            mlflow.log_param('x_shape', str(x.shape))
            mlflow.log_param('helper', helper)

            scores = []
            for i, (trn, val) in enumerate(folds):
                fold_scores = []
                print(i + 1, 'fold.')

                trn_x = x.iloc[trn,:].values
                trn_y = y.iloc[trn,:].values
                val_x = x.iloc[val,:].values
                val_y = y.iloc[val,:].values
                if helper == 'both':
                    p100_model = load_model('./p100/' + cell_line)
                    gcp_model = load_model('./gcp/' + cell_line)
                    p100_trx = p100_model.predict(trn_x)
                    p100_vlx = p100_model.predict(val_x)
                    p100_trx = p100_trx.astype(np.float)
                    p100_vlx = p100_vlx.astype(np.float)

                    trn_x_ = np.concatenate((trn_x, p100_trx), axis=1)
                    val_x_ = np.concatenate((val_x, p100_vlx), axis=1)
                    gcp_trx = gcp_model.predict(trn_x)
                    gcp_vlx = gcp_model.predict(val_x)

                    gcp_trx = gcp_trx.astype(np.float)
                    gcp_vlx = gcp_vlx.astype(np.float)

                    trn_x = np.concatenate((trn_x_, gcp_trx), axis=1)
                    val_x = np.concatenate((val_x_, gcp_vlx), axis=1)
                else:
                    helper_model = load_model('./' + helper + '/' + cell_line + '/')
                    trn_x = trn_x.astype(np.float)
                    val_x = val_x.astype(np.float)
                    _trx = helper_model.predict(trn_x)
                    _vlx = helper_model.predict(val_x)

                    _trx = _trx.astype(np.float)
                    _vlx = _vlx.astype(np.float)

                    trn_x = np.concatenate((trn_x, _trx), axis=1)
                    val_x = np.concatenate((val_x, _vlx), axis=1)
                    pd.DataFrame(trn_x).to_csv('trn.csv')

                if descriptor == 'jtvae':
                    scaler = StandardScaler().fit(trn_x)
                    trn_x = scaler.transform(trn_x)
                    val_x = scaler.transform(val_x)

                trn_y_list = []
                val_y_list = []

                for j in range(trn_y.shape[1]):
                    trn_y_list.append(trn_y[:,j])
                    val_y_list.append(val_y[:,j])

                print('Train shapes:', trn_x.shape, trn_y.shape)
                print('Test shapes:', val_x.shape, val_y.shape)

                weights = []
                for trn_target in trn_y_list:
                    w = class_weight.compute_class_weight('balanced', np.unique(trn_target), trn_target)
                    weights.append({0: w[0], 1: w[1]})

                model = build_model(trn_x.shape[1], trn_y.shape[1])
                model.fit(trn_x, trn_y_list,
                          class_weight=weights, batch_size=16, epochs=100, verbose=1)

                mlflow.keras.log_model(model, 'model_fold_' + str(i + 1))
                print('Model saved in run %s' % mlflow.active_run().info.run_uuid)

                for i in range(trn_y.shape[1]):
                    if trn_y.shape[1] == 1:
                        pred = model.predict(val_x)
                    else:
                        pred = model.predict(val_x)[i]
                    fold_scores.append(roc_auc_score(val_y_list[i], pred))

                scores.append(fold_scores)
                K.clear_session()

            scores = np.asanyarray(scores)
            mean_auc = np.mean(scores, axis=0)

            for idx in range(len(indexes)):
                mlflow.log_metric('Gene_' + str(indexes[idx]) + '_AUC', mean_auc[idx])

            ma.log_artifact(mlflow.get_artifact_uri())
            mlflow.end_run()
