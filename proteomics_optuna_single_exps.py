from keras.layers import Dense, Dropout, Activation, BatchNormalization, Input
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.models import load_model
from keras import backend as K
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from get_data import GetData
import autokeras as ak
import json
import pandas as pd
import numpy as np
import mlflow.keras
import mlflow
import model_artifact as ma
import autokeras as ak
import optuna
import os

cell_line = 'A549'
target = 'dn'

with open('./L1000CDS_subset.json', 'r') as f:
    L = json.load(f)


def build_model(input_dim):
    input_layer = Input(shape=(input_dim,))
    _ = Dense(units=256)(input_layer)
    _ = BatchNormalization()(_)
    _ = Activation('relu')(_)
    _ = Dropout(0.7)(_)
    _ = Dense(units=32)(_)
    output_layer = Dense(units=1, activation='sigmoid')(_)

    model = Model(inputs=input_layer, outputs=output_layer)
    opt = SGD(learning_rate=0.001)
    model.compile(optimizer=opt, loss='binary_crossentropy')

    return model

def create_model(n_layer, activation, mid_units, dropout_rate, input_dim):
    model = Sequential()

    model.add(Input(shape=(input_dim,)))
    for i in range(n_layer):
        model.add(Dense(32,activation=activation))
        model.add(BatchNormalization())
    model.add(Dense(mid_units, activation=activation))
    model.add(Dropout(dropout_rate))

    model.add(Dense(units=1))

    return model

def objective(trial):
    descriptor = 'jtvae'
    global cell_line
    global target
    #mlflow.set_tracking_uri('http://193.140.108.166:5000')
    mlflow.set_experiment('Optuna_' + cell_line + '_experiments_min_100_')
    mlflow.start_run()
    mlflow.set_tag('Author', 'Bahattin')
    obj = GetData(L=L, cell_line=cell_line, descriptor=descriptor, n_fold=5, random_state=42,
                  random_genes=False, csv_file='JTVAE_Representations.csv')
    mlflow.log_param('Model_Type', 'Single-Task-NN')
    if target == 'up':
        mlflow.log_param('target', 'up_genes')
        x, y, folds = obj.get_up_genes()
    elif target == 'dn':
        mlflow.log_param('target', 'down_genes')
        x, y, folds = obj.get_down_genes()

    lst_y = []
    for i in range(978):
        lst_y.append(np.count_nonzero(y.iloc[:,i]))
    df = pd.DataFrame({'genes': lst_y})
    indexes = df[df['genes'] >= 100].index.values
    del lst_y, df
    y = y[y.columns[indexes]]
    x.drop(['SMILES'], axis=1, inplace=True)

#                 mlflow.log_param('cell_line', cell_line)
#                 mlflow.log_param('descriptor', descriptor)
#                 mlflow.log_param('n_fold', '5')
#                 mlflow.log_param('random_state', '42')
#                 mlflow.log_param('batch_size', '16')
#                 mlflow.log_param('epochs', '50')
#                 mlflow.log_param('optimizer', 'adam')
#                 mlflow.log_param('loss', 'binary_crossentropy')
#                 mlflow.log_param('early_stopping', 'True')
#                 mlflow.log_param('x_shape', str(x.shape))
#                 mlflow.log_param('extra_feature', helper)

    n_layer = trial.suggest_int('n_layer', 1, 5)
    mid_units = int(trial.suggest_discrete_uniform('mid_units', 50, 500, 1))
    dropout_rate = trial.suggest_uniform('dropout_rate', 0, 1)
    activation = trial.suggest_categorical('activation', ['relu', 'sigmoid'])
    optimizer = trial.suggest_categorical('optimizer', ['sgd', 'adam', 'rmsprop'])
    helper = trial.suggest_categorical('helper', ['P100', 'GCP', 'both'])
    epochs = trial.suggest_int('epochs', 10,200,10)
    scores = []

    mlflow.log_param('cell_line', cell_line)
    mlflow.log_param('n_layer', n_layer)
    mlflow.log_param('mid_units', mid_units)
    mlflow.log_param('dropout_rate', dropout_rate)
    mlflow.log_param('activation', activation)
    mlflow.log_param('optimizer', optimizer)
    mlflow.log_param('helper', helper)
    mlflow.log_param('epochs', epochs)

    for idx in range(len(indexes)):
        folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(x, y.iloc[:,idx]))
        fold_scores = []

        for i, (trn, val) in enumerate(folds):
            print(i+1, 'fold.')

            trn_x = x.iloc[trn,:].values
            trn_y = y.iloc[trn,idx].values
            val_x = x.iloc[val,:].values
            val_y = y.iloc[val,idx].values


            if descriptor == 'jtvae':
                scaler = StandardScaler().fit(trn_x)
                trn_x = scaler.transform(trn_x)
                val_x = scaler.transform(val_x)

            if helper == 'both':
                p100_model = load_model('./P100/' + cell_line + '/model_autokeras', custom_objects=ak.CUSTOM_OBJECTS)
                gcp_model = load_model('./GCP/' + cell_line + '/model_autokeras', custom_objects=ak.CUSTOM_OBJECTS)
                trn_x = trn_x.astype(np.float)
                val_x = val_x.astype(np.float)
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
                helper_model = load_model('./' + helper + '/' + cell_line + '/' + 'model_autokeras', custom_objects=ak.CUSTOM_OBJECTS)
                trn_x = trn_x.astype(np.float)
                val_x = val_x.astype(np.float)
                _trx = helper_model.predict(trn_x)
                _vlx = helper_model.predict(val_x)

                _trx = _trx.astype(np.float)
                _vlx = _vlx.astype(np.float)

                trn_x = np.concatenate((trn_x, _trx), axis=1)
                val_x = np.concatenate((val_x, _vlx), axis=1)

            print('Train shapes:', trn_x.shape, trn_y.shape)
            print('Test shapes:', val_x.shape, val_y.shape)

            w = class_weight.compute_class_weight('balanced', np.unique(trn_y), trn_y)
            weights = {0: w[0], 1: w[1]}


            #model = build_model(trn_x.shape[1])
            #model.fit(trn_x, trn_y, validation_data=(val_x, val_y),
                      #class_weight=weights, batch_size=16, epochs=50, verbose=1)

            model = create_model(n_layer, activation, mid_units, dropout_rate,trn_x.shape[1])
            model.compile(optimizer=optimizer,
                    loss='mean_squared_error',
                    metrics=['mae'])
            history = model.fit(trn_x, trn_y,
            verbose=100,
            epochs=epochs,
            validation_data=(val_x, val_y),batch_size=32)

            #clf = model
           # clf.save("Optuna_/" + cell_line + '/' + helper + '/' +'model_fold_' + str(i + 1) + '_' + str(indexes[idx]))
            # mlflow.keras.log_model(model, 'model_fold_' + str(i + 1) + '_' + str(indexes[idx]))
            # print('Model saved in run %s' % mlflow.active_run().info.run_uuid)

            pred = model.predict(val_x)
            fold_scores.append(roc_auc_score(val_y, pred))
            K.clear_session()
        scores.append(np.mean(np.asanyarray(fold_scores), axis=0))

        mlflow.log_metric('Gene_' + str(indexes[idx]) + '_AUC', np.mean(np.asanyarray(fold_scores), axis=0))

    scores = pd.DataFrame(scores)
    if not os.path.isfile(helper+'_'+cell_line+'_'+target+'.csv'):
        scores.to_csv(helper+'_'+cell_line+'_'+target+'.csv')
    else:
        scores.to_csv(helper+'_'+cell_line+'_'+target+'.csv', mode='a', header=False)
    mean_auc = np.mean(scores, axis=0)
    mlflow.end_run()
    return mean_auc

def main():
    global cell_line
    global target
    for cell_line in [ 'MCF7', 'PC3']:
        for target in ['up', 'dn']:
            if(cell_line == 'MCF7' and target == 'up'):
                continue
            study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize')
            study.optimize(objective, n_trials=100)
            print('best_params')
            print(study.best_params)
            print('-1 x best_value')
            print(-study.best_value)

            print('\n --- sorted --- \n')
            sorted_best_params = sorted(study.best_params.items(), key=lambda x : x[0])
            for i, k in sorted_best_params:
                print(i + ' : ' + str(k))


if __name__ == '__main__':
    main()
