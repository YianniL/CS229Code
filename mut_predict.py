import numpy as np
import csv
import argparse
import dill
import itertools

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def train_logistic(X_train, y_train, X_val=None, y_val=None, hyp=None):
    best_c = None    
    best_model = None
    best_auc = 0.0
    if hyp is None:
        hyp = [1.0]
    for c in hyp:
        logistic = LogisticRegression(random_state=23, solver='lbfgs', C=c).fit(X_train, y_train)
        auc = roc_auc_score(y_val, logistic.predict_proba(X_val)[:, 1]) #grab the positive probs
        if auc > best_auc:
            best_c = c
            best_auc = auc
            best_model = logistic
    print 'Best C: {}'.format(best_c)
    # print 'Best AUC: {}'.format(best_auc)
    # test_model(X_train, y_train, best_model)
    return best_model

def train_ensemble(X_train, y_train, models, X_val=None, y_val=None, hyp=None):
    best_c = None
    best_model = None
    best_auc = 0.0
    if hyp is None:
        hyp = [1.0]
    for c in hyp:
        preds_train = []
        preds_val = []
        for m in models:
            proba_op = getattr(m, 'predict_proba', None)
            if callable(proba_op):
                preds_train.append(np.expand_dims(m.predict_proba(X_train)[:, 1], axis=1))
                preds_val.append(np.expand_dims(m.predict_proba(X_val)[:, 1], axis=1))
            else:
                preds_train.append(np.expand_dims(m.decision_function(X_train), axis=1))
                preds_val.append(np.expand_dims(m.decision_function(X_val), axis=1))
        ensemble_X_train = np.concatenate(preds_train, axis=1)
        ensemble_X_val = np.concatenate(preds_val, axis=1)
        ensemble_logistic = LogisticRegression(random_state=23, solver='lbfgs', C=c).fit(ensemble_X_train, y_train)
        auc = roc_auc_score(y_val, ensemble_logistic.predict_proba(ensemble_X_val)[:, 1]) #grab the positive probs
        if auc > best_auc:
            best_c = c
            best_auc = auc
            best_model = ensemble_logistic
    print 'Best C: {}'.format(best_c)
    print 'Best AUC: {}'.format(best_auc)
    return best_model

def train_svm(X_train, y_train, X_val=None, y_val=None, C=None, gamma=None):
    best_c = None
    best_gamma = None    
    best_model = None
    best_auc = 0.0
    if C is None:
        C = [1.0]
    if gamma is None:
        gamma = ['auto']
    for c in C:
        for g in gamma:
            svm = SVC(C=c, gamma=g, kernel='rbf').fit(X_train, y_train)
            auc = roc_auc_score(y_val, svm.decision_function(X_val))
            if auc > best_auc:
                best_c = c
                best_gamma = g
                best_model = svm
                best_auc = auc
    print 'Best C: {}'.format(best_c)
    # print 'Best Gamma: {}'.format(best_gamma)
    # print 'Best AUC: {}'.format(best_auc)
    test_model(X_train, y_train, best_model)
    return best_model

def train_rf(X_train, y_train, X_val=None, y_val=None, num_estimators=None, max_features=None):
    best_numest = None
    best_maxfeat = None
    best_model = None
    best_auc = 0.0
    if num_estimators is None:
        num_estimators = [10]
    else:
        num_estimators = [int(i) for i in num_estimators]
    if max_features is None:
        max_features = ['auto']
    # else:
        # max_features = ['auto'] + [i in max_features if i < X_val.shape[1]]
    for ne in num_estimators:
        for mf in max_features:
            rf = RandomForestClassifier(random_state=23, n_estimators=ne, max_features=mf).fit(X_train, y_train)
            auc = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])
            if auc > best_auc:
                best_numest = ne
                best_maxfeat = mf
                best_auc = auc
                best_model = rf
    print 'Best NumEst: {}'.format(best_numest)
    print 'Best MaxFeat: {}'.format(best_maxfeat)
    # print 'Best AUC: {}'.format(best_auc)
    test_model(X_train, y_train, best_model)
    return best_model

def test_model(X, y, model, dg):
    preds = model.predict(X)
    acc = model.score(X, y)
    auc_op = getattr(model, 'predict_proba', None)
    if callable(auc_op):
        auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    else:
        auc = roc_auc_score(y, model.decision_function(X))
    print 'Accuracy: {}, AUC: {}'.format(acc, auc)
    cm = confusion_matrix(y, preds)
    print cm
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    print tn, fp, fn, tp
    print 'sensitivity is {}'.format((1.0*tp)/(tp + fn))
    print 'specificity is {}'.format((1.0*tn)/(tn + fp))
    print '\n'

    tn_dg = 0.0
    fp_dg = 0.0
    fn_dg = 0.0
    tp_dg = 0.0

    counter = 0.0

    for i in range(dg.shape[0]):
        if preds[i] == y[i] and preds[i] == 0:
            tn_dg += abs(dg[i])
        elif preds[i] != y[i] and preds[i] == 1:
            fp_dg += abs(dg[i])
        elif preds[i] != y[i] and preds[i] == 0:
            fn_dg += abs(dg[i])
        else:
            tp_dg += abs(dg[i])

    plot_confusion_matrix(cm, ['Nullspot', 'Hotspot'],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues)

    print tn_dg / tn
    print fp_dg / fp
    print fn_dg / fn
    print tp_dg / tp

    # fpr, tpr, thresholds = roc_curve(y, model.predict_proba(X)[:, 1])
    # plt.title('ROC Curve')
    # plt.plot(fpr, tpr, 'b')
    # # plt.legend(loc = 'lower right')
    # plt.plot([0, 1], [0, 1],'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.savefig('roccurveworse.png')

    return preds

def test_ensemble_model(X, y, ensemble_logistic, models):
    preds_test = []
    for m in models:
        if callable(getattr(m, 'predict_proba', None)):
            preds_test.append(np.expand_dims(m.predict_proba(X)[:, 1], axis=1))
        else:
            preds_test.append(np.expand_dims(m.decision_function(X), axis=1))
    ensemble_X_test = np.concatenate(preds_test, axis=1)
    preds = ensemble_logistic.predict(ensemble_X_test)
    acc = ensemble_logistic.score(ensemble_X_test, y)
    auc = roc_auc_score(y, ensemble_logistic.predict_proba(ensemble_X_test)[:, 1])
    print 'Accuracy: {}, AUC: {}'.format(acc, auc)
    return preds

def load_sasnet_features(path):
    X = np.loadtxt(open(path, "rb"), delimiter=",")
    y = np.loadtxt(open(path.replace('_x', '_y'), "rb"), delimiter=",")
    print 'Loaded from path {}'.format(path)
    print 'X shape {} y shape {}'.format(X.shape, y.shape)
    return X, y

def load_moreira_features(path):
    X, y = [], []
    with open(path, "rb") as f:
        reader = csv.reader(f)
        rows = []
        for row in reader:
            rows.append(row)
        for i in range(1, len(rows)):
            row = rows[i]
            if row[5].strip() == 'HS':
                y.append(1)
            else:
                y.append(0)
            X.append([float(d) for d in row[6:]])
    X = np.asarray(X)
    y = np.asarray(y)
    print 'Loaded from path {}'.format(path)
    print 'X shape {} y shape {}'.format(X.shape, y.shape)
    return X, y

def load_delta_G(path):
    dg = []
    with open(path, 'rb') as f:
        reader = csv.reader(f)
        rows = []
        for row in reader:
            rows.append(row)
        for i in range(1, len(rows)):
            row = rows[i]
            if row[4] == 'NS':
                dg.append(0.0)
            elif row[4] == 'HS':
                dg.append(2.0)
            else:
                dg.append(float(row[4]))
    return np.asarray(dg)

def load_conv_features(conv_path, moreira_path):
    print conv_path
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
    for d in data:
        X.append(raw_feat[d][1])
    X = np.asarray(X)
    y = np.asarray(y)
    print 'Loaded conv features from path {}'.format(conv_path)
    print 'X shape {} y shape {}'.format(X.shape, y.shape)
    return X, y

#Note, taken from sklearn docs
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion.png')
    plt.close()

def preprocess(X, y, p, normalize=True, scale_up=True):
    X = X[p]
    y = y[p]

    X_train = X[:int(X.shape[0]*.6), :]
    y_train = y[:int(X.shape[0]*.6)]

    X_val = X[int(X.shape[0]*.6): int(X.shape[0]*.8), :]
    y_val = y[int(X.shape[0]*.6):int(X.shape[0]*.8)]

    X_test = X[int(.8*X.shape[0]):, :]
    y_test = y[int(.8*X.shape[0]):]

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
        X_train = X_train[p2]
        y_train = y_train[p2]
        print 'pos is {}, shape is {}\n\n\n'.format(np.sum(y_train), y_train.shape)

    print('Train shape {} Val shape {} Test shape {}').format(X_train.shape[0], X_val.shape[0], X_test.shape[0])
    return X_train, y_train, X_val, y_val, X_test, y_test

def main():
    parser = argparse.ArgumentParser(description='Parser for training/evaluating hotspot prediction models.')
    parser.add_argument('features', help='SASnet features, moreira features, or both')
    parser.add_argument('model', help='which prediction algorithm to use')
    parser.add_argument('--save', help='where to save things', type=bool)
    parser.add_argument('--hyp1', help='1st hyperparameter', type=str)
    parser.add_argument('--hyp2', help='2nd hyperparamter', type=str)
    args = parser.parse_args()

    np.random.seed(23)

    moreira_path = '/home/users/jlalouda/surfacelets/hotspot_corrected.csv'
    yianni_path = '/scratch/PI/rondror/jlalouda/hotspot_x.csv'
    conv_path  = '/scratch/PI/rondror/jlalouda/hotspot_saved_grids_and_convs/ex.dill'
    
    X, y, p = None, None, None
    if args.features == 'sasnet':
        X, y = load_sasnet_features(yianni_path)
    elif args.features == 'moreira':
        X, y = load_moreira_features(moreira_path)
    elif args.features == 'combine_sasmor':
        X_y, y_y = load_sasnet_features(yianni_path) #concatenate the stuff here
        X_m, y_m = load_moreira_features(moreira_path)
        X = np.concatenate([X_m, X_y[:, 20:]], axis=1)
        y = y_y
    elif args.features == 'conv':
        X, y = load_conv_features(conv_path, moreira_path)
    elif args.features == 'combine_convsasmor':
        X_y, y_y = load_sasnet_features(yianni_path) #concatenate the stuff here
        X_m, y_m = load_moreira_features(moreira_path)
        X_c, y_c = load_conv_features(conv_path, moreira_path)
        X = np.concatenate([X_m, X_y[:, 20:], X_c], axis=1)
        y = y_y
    elif args.features == 'roc':
        pass
    else:
        print 'Unrecognized option for features'
        return

    p = np.random.permutation(X.shape[0])
    
    # X_train, y_train, X_val, y_val, X_test, y_test = preprocess(X, y, p, normalize=False, scale_up=False)
    # with open('/home/users/jlalouda/surfacelets/src/mutation/test_y.npy', 'w') as f:
    #     np.save(f, y_test)
    # with open('/home/users/jlalouda/surfacelets/src/mutation/val_y.npy', 'w') as f:
    #     np.save(f, y_val)

    # if args.model != 'ensemble':
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess(X, y, p, normalize=True, scale_up=True)
    dg = load_delta_G(moreira_path)
    dg = dg[p]
    test_dg = dg[int(.8*dg.shape[0]):]
    print(np.shape(y_train), np.sum(y_train))
    print(np.shape(y_val), np.sum(y_val))
    print(np.shape(y_test), np.sum(y_test))

    # else:
    #     models = ['logistic', 'svm', 'rf']

    #     with open('/home/users/jlalouda/surfacelets/src/mutation/test_y.npy', 'r') as f:
    #         y_test = np.load(f)
    #     with open('/home/users/jlalouda/surfacelets/src/mutation/test_y.npy', 'r') as f:
    #         y_val = np.load(f)
    #     for m in models:
    #         filename = '/home/users/jlalouda/surfacelets/src/mutation/predictions_' + 'val_' + args.model + '_' + args.features + '.npy'
    #         with open(filename, 'r') as f:
    #             temp.append(np.load(f))
    #     y_test
    
    hyp1, hyp2 = None, None  
    if args.hyp1 is not None:
        hyp1 = [float(i) for i in args.hyp1.split(',')]
        # if len(hyp1) == 2:
        #     hyp1 = np.random.uniform(hyp1[0], hyp1[1], 1000).tolist()
    if args.hyp2  is not None:
        hyp2 = [float(i) for i in args.hyp2.split(',')]

    preds = None
    if args.model == 'logistic':
        logistic = train_logistic(X_train, y_train, X_val=X_val, y_val=y_val, hyp=hyp1)
        preds = test_model(X_test, y_test, logistic, test_dg)
    elif args.model == 'svm':
        svm = train_svm(X_train, y_train, X_val=X_val, y_val=y_val, C=hyp1, gamma=hyp2)
        preds = test_model(X_test, y_test, svm)
    elif args.model == 'rf':
        rf = train_rf(X_train, y_train, X_val=X_val, y_val=y_val, num_estimators=hyp1, max_features=hyp2)
        preds = test_model(X_test, y_test, rf)
    elif args.model == 'ensemble':
        log_c = [001,.002,.003,.004,.005,.006,.007,.005,.1,.2,.3,.4,1]
        logistic = train_logistic(X_train, y_train, X_val=X_val, y_val=y_val, hyp=log_c)
        svm_c = [.1,1,10,20,50,60,70,80,100]
        svm_g = [.0001,.001,.01,.1,1,10,100]
        svm = train_svm(X_train, y_train, X_val=X_val, y_val=y_val, C=svm_c, gamma=svm_g)
        rf_ne = [3, 5, 10, 100, 1000]
        rf = train_rf(X_train, y_train, X_val=X_val, y_val=y_val, num_estimators=rf_ne, max_features=None)
        ensemble = train_ensemble(X_train, y_train, [logistic, svm, rf], X_val=X_val, y_val=y_val, hyp=hyp1)
        preds = test_ensemble_model(X_test, y_test, ensemble, [logistic, svm, rf])
    elif args.model == 'roc':
        X, y = load_moreira_features(moreira_path)
        p = np.random.permutation(X.shape[0])
        X_train, y_train, X_val, y_val, X_test, y_test = preprocess(X, y, p, normalize=True, scale_up=True)
        logistic1 = train_logistic(X_train, y_train, X_val=X_val, y_val=y_val, hyp=hyp1)
        fpr1, tpr1, thresholds1 = roc_curve(y_test, logistic1.predict_proba(X_test)[:, 1])
        
        X_y, y_y = load_sasnet_features(yianni_path) #concatenate the stuff here
        X_m, y_m = load_moreira_features(moreira_path)
        X = np.concatenate([X_m, X_y[:, 20:]], axis=1)
        y = y_y
        X_train, y_train, X_val, y_val, X_test, y_test = preprocess(X, y, p, normalize=True, scale_up=True)
        logistic2 = train_logistic(X_train, y_train, X_val=X_val, y_val=y_val, hyp=hyp1)
        fpr2, tpr2, thresholds2 = roc_curve(y_test, logistic2.predict_proba(X_test)[:, 1])

        plt.title('ROC Curve')
        plt.plot(fpr1, tpr1, 'b', label='Moreira Features')
        plt.plot(fpr2, tpr2, 'r', label='Combined Features')
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'k--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig('roccurve_combined.png')

    else:
        print 'model not recognized'
        return
    if args.save:
        filename = '/home/users/jlalouda/surfacelets/src/mutation/predictions_' + args.model + '_' + args.features + '.npy'
        with open(filename, 'w') as f:
            np.save(f, preds)

main()