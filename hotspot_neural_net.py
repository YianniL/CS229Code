import numpy as np
import csv
import argparse
import dill
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

def gen_from_indices(indices, data, raw_feat, X, y, expand_y=False):
    X_new = []
    y_new = []
    for i in range(indices.shape[0]):
        index = indices[i]
        key = data[index]
        for j in range(raw_feat[key][1].shape[0]):
            X_new.append(raw_feat[key][1][j])
            if expand_y:
                y_new.append(y[index])
        if not expand_y:
            y_new.append(y[index])
    return np.asarray(X_new), np.asarray(y_new)

def load_full_conv_feat(moreira_path, conv_path, normalize=True, scale_up=True):
    np.random.seed(23)
    with open(conv_path, 'rb') as f:
        raw_feat = dill.load(f)
    X, y, data = [], [], []
    with open(moreira_path, "rb") as f:
        reader = csv.reader(f)
        rows = []
        for row in reader:
            rows.append(row)
        currProt = None
        for i in range(1, len(rows)):
            row = rows[i]
            if row[0] != currProt and row[0] != '':
                currProt = row[0]
            data.append((currProt, row[1], row[2]))
            if row[5].strip() == 'HS':
                y.append(1)
            else:
                y.append(0)
    p = np.random.permutation(len(data))
    indices = np.arange(len(data))
    indices = indices[p]

    train_indices = indices[:int(len(y)*.6)]

    val_indices = indices[int(len(y)*.6): int(len(y)*.8)]

    test_indices = indices[int(.8*len(y)):]

    X_train, y_train = gen_from_indices(train_indices, data, raw_feat, X, y, expand_y=True)
    X_val, y_val = gen_from_indices(val_indices, data, raw_feat, X, y)
    X_test, y_test = gen_from_indices(test_indices, data, raw_feat, X, y)

    print X_train.shape, y_train.shape
    print X_val.shape, y_val.shape
    print X_test.shape, y_test.shape

    if normalize:
        eps = 1e-5
        mu = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0) + eps #Deal with dividing by 0
        X_train = (X_train - mu) / std
        X_val = (X_val - mu) / std
        X_test = (X_test - mu) / std

    if scale_up:
        num_pos = np.sum(y_train)
        num_neg = y_train.shape[0] - num_pos
        counter = abs(num_neg - num_pos)
        X_train = X_train.tolist()
        y_train = y_train.tolist()
        lesser = 1 if num_pos < num_neg else 0
        new_x, new_y = [], []
        while counter > 0:
            for i in range(len(y_train)):
                if int(y_train[i]) == lesser and counter > 0:
                    new_x.append(X_train[i])
                    new_y.append(y_train[i])
                    counter -= 1
        X_train += new_x
        y_train += new_y
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        p2 = np.random.permutation(X_train.shape[0])
        X_train_scram = X_train[p2]
        y_train_scram = y_train[p2]

    print X_train.shape, y_train.shape
    print X_val.shape, y_val.shape
    print X_test.shape, y_test.shape

    print 'Loaded conv features from path {}'.format(conv_path)
    return X_train_scram, X_val, X_test, y_train_scram, y_val, y_test, X_train, y_train

def consensus_pred(preds, num_rot=20):
    # print preds.shape[0]/num_rot
    # print 'in here'
    con_pred = np.zeros(preds.shape[0]/num_rot)
    for i in range(0, preds.shape[0], num_rot):
        # print 'doop'
        # print np.mean(preds[i:i+num_rot])
        con_pred[i/20] = np.mean(preds[i:i+num_rot])
    # print con_pred.shape
    act_preds = np.zeros(con_pred.shape[0])
    act_preds[con_pred > .5] = 1
    # print act_preds.shape
    return act_preds, con_pred

def get_acc(preds, labels):
    return (preds == labels).mean()

def train(X_train, y_train, X_val, y_val, X_train_reg, y_train_reg):
    inputs = keras.Input(shape=(512,))
    # noise = keras.layers.GaussianNoise(.4)(inputs)
    l1 = keras.layers.Dense(20, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001))(inputs)
    l2 = keras.layers.Dropout(0.25)(l1, training=True)
    l3 = keras.layers.Dense(10, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001))(l2)
    l4 = keras.layers.Dropout(0.25)(l3, training=True)
    l5 = keras.layers.Dense(1, activation=tf.nn.sigmoid)(l4)
    model = keras.Model(inputs, l5)
    model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='binary_crossentropy',
              metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=15)

    train_preds = model.predict(X_train_reg)
    train_preds, train_probs = consensus_pred(train_preds)
    y_train_reg = np.asarray([y_train_reg[i] for i in range(0, len(y_train_reg), 20)])
    print get_acc(train_preds, y_train_reg)
    print roc_auc_score(y_train_reg, train_probs)
    tn, fp, fn, tp = confusion_matrix(y_train_reg, train_preds).ravel()
    print tn, fp, fn, tp
    print 'sensitivity is {}'.format((1.0*tp)/(tp + fn))
    print 'specificity is {}'.format((1.0*tn)/(tn + fp))
    print '\n'



    # val_loss, val_acc = model.evaluate(X_val, y_val)
    val_preds = model.predict(X_val)
    val_preds, val_probs = consensus_pred(val_preds)
    print val_preds == y_val
    print get_acc(val_preds, y_val)
    auc = roc_auc_score(y_val, val_probs)
    print confusion_matrix(y_val, val_preds)
    print auc
    model_json = model.to_json()
    with open("/scratch/PI/rondror/jlalouda/hotspot_neural_net_model/test1.json", "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("/scratch/PI/rondror/jlalouda/hotspot_neural_net_model/model_test1.h5")
        print("Saved model to disk")

def test(X_test, y_test):
    print 'in test'
    json_file = open("/scratch/PI/rondror/jlalouda/hotspot_neural_net_model/test1.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights("/scratch/PI/rondror/jlalouda/hotspot_neural_net_model/model_test1.h5")
    print("Loaded model from disk")
    loaded_model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='binary_crossentropy',
              metrics=['accuracy'])
    test_preds = loaded_model.predict(X_test)
    test_preds, test_probs = consensus_pred(test_preds)
    print confusion_matrix(y_test, test_preds)
    print get_acc(test_preds, y_test)
    print roc_auc_score(y_test, test_probs)
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    print tn, fp, fn, tp
    print 'sensitivity is {}'.format((1.0*tp)/(tp + fn))
    print 'specificity is {}'.format((1.0*tn)/(tn + fp))
    print '\n'

def main():
    parser = argparse.ArgumentParser(description='neural net parser')
    parser.add_argument('--train', dest='train', default=False, action='store_true', help='are we training?')
    parser.add_argument('--save', help='where to save things', type=bool)
    args = parser.parse_args()

    np.random.seed(23)

    moreira_path = '/home/users/jlalouda/surfacelets/hotspot_corrected.csv'
    conv_path  = '/scratch/PI/rondror/jlalouda/hotspot_saved_grids_and_convs/ex_20.dill'

    X_train, X_val, X_test, y_train, y_val, y_test, X_train_reg, y_train_reg = load_full_conv_feat(moreira_path, conv_path, normalize=True, scale_up=True)
    
    if args.train:
        train(X_train, y_train, X_val, y_val, X_train_reg, y_train_reg)
    else:
        test(X_test, y_test)

main()
