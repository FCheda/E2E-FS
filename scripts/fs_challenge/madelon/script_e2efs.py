from keras.utils import to_categorical
from keras import callbacks, regularizers
import json
import numpy as np
import os
from dataset_reader import madelon
from src.layers import e2efs
from src.utils import balance_accuracy
from src.svc.models import LinearSVC
from extern.liblinear.python import liblinearutil
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import average_precision_score
from keras import backend as K
from src import callbacks as clbks, optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


batch_size = 2
epochs = 150
extra_epochs = 200
mu = 100
kernel = 'linear'
reps = 1
verbose = 0
loss_function = 'square_hinge'
k_folds = 3
k_fold_reps = 20
optimizer_class = optimizers.E2EFS_Adam
normalization_func = madelon.Normalize

dataset_name = 'madelon'
directory = os.path.dirname(os.path.realpath(__file__)) + '/info/'
e2efs_classes = [
    (e2efs.E2EFSSoft, {'dropout': .1, 'decay_factor': .75, 'kernel_regularizer': regularizers.l2(1e-2)}),
    (e2efs.E2EFSHard, {'dropout': .1, 'l1': 1e0, 'l2': 1e0}),
]


def e2efs_factor(T=250):
    def func(epoch):
        if epoch < 5:
            return 0., 0., .5
        elif epoch < extra_epochs:
            return 1., min(1., (epoch - 5) / T), .5
        else:
            return 1., 1., 0.
    return func


def e2efs_units(units, input_shape, T):
    p = (units / input_shape) ** (1. / T)

    def func(epoch):
        return max(units, input_shape*p**epoch)

    return func


def scheduler(extra=0, factor=.1):
    def sch(epoch):
        if epoch < extra:
            return .1 * factor
        elif epoch < 50 + extra:
            return .01 * factor
        elif epoch < 100 + extra:
            return .002 * factor
        else:
            return .0004 * factor
    return sch


def load_dataset():
    dataset = madelon.load_dataset()
    return dataset


def train_Keras(train_X, train_y, test_X, test_y, kwargs, e2efs_class=None, n_features=None, e2efs_kwargs=None):
    normalization = normalization_func()
    num_classes = train_y.shape[-1]

    norm_train_X = normalization.fit_transform(train_X)
    norm_test_X = normalization.transform(test_X)

    batch_size = max(2, len(train_X) // 50)
    class_weight = train_y.shape[0] / np.sum(train_y, axis=0)
    class_weight = num_classes * class_weight / class_weight.sum()
    sample_weight = None
    print('reps : ', reps, ', weights : ', class_weight)
    if num_classes == 2:
        sample_weight = np.zeros((len(norm_train_X),))
        sample_weight[train_y[:, 1] == 1] = class_weight[1]
        sample_weight[train_y[:, 1] == 0] = class_weight[0]
        class_weight = None

    svc_model = LinearSVC(nfeatures=norm_train_X.shape[1:], **kwargs)
    svc_model.create_keras_model(nclasses=num_classes)

    model_clbks = [
        callbacks.LearningRateScheduler(scheduler(extra=0 if e2efs_class is None else extra_epochs)),
    ]

    if e2efs_class is not None:
        classifier = svc_model.model
        e2efs_layer = e2efs_class(n_features, input_shape=norm_train_X.shape[1:], **e2efs_kwargs)
        model = e2efs_layer.add_to_model(classifier, input_shape=norm_train_X.shape[1:])
        model_clbks.append(
            clbks.E2EFSCallback(factor_func=e2efs_factor(),
                                units_func=e2efs_units(
                                    n_features, norm_train_X.shape[-1], extra_epochs
                                ) if 'hard' in e2efs_class.__name__.lower() else None,
                                verbose=verbose)
        )
    else:
        model = svc_model.model
        e2efs_layer = None

    optimizer = optimizer_class(e2efs_layer, lr=1e-3)
    model_epochs = epochs
    if e2efs_class is not None:
        model_epochs += extra_epochs

    model.compile(
        loss=LinearSVC.loss_function(loss_function, class_weight),
        optimizer=optimizer,
        metrics=[LinearSVC.accuracy]
    )

    if e2efs_class is not None:
        model.fs_layer = e2efs_layer
        model.heatmap = e2efs_layer.moving_heatmap

    model.fit(
        norm_train_X, train_y, batch_size=batch_size,
        epochs=model_epochs,
        callbacks=model_clbks,
        validation_data=(norm_test_X, test_y),
        class_weight=class_weight,
        sample_weight=sample_weight,
        verbose=verbose
    )

    model.normalization = normalization

    return model


def train_SVC(train_X, train_y, kwargs):
    params = '-q -s ' + str(kwargs['solver']) + ' -c ' + str(kwargs['C'])
    model = liblinearutil.train((2 * train_y[:, -1] - 1).tolist(), train_X.tolist(), params)
    return model


def main(dataset_name):

    dataset = load_dataset()

    raw_data = np.asarray(dataset['raw']['data'])
    raw_label = np.asarray(dataset['raw']['label'])
    num_classes = len(np.unique(raw_label))

    rskf = RepeatedStratifiedKFold(n_splits=k_folds, n_repeats=k_fold_reps, random_state=42)

    for e2efs_class, e2efs_kwargs in e2efs_classes:
        print('E2EFS-Method : ', e2efs_class.__name__)

        nfeats = []
        accuracies = []
        model_accuracies = []
        svc_accuracies = []
        BAs = []
        svc_BAs = []
        model_BAs = []
        mAPs = []
        svc_mAPs = []
        model_mAPs = []
        mus = []
        name = dataset_name + '_' + kernel + '_mu_' + str(mu)
        print(name)

        for j, (train_index, test_index) in enumerate(rskf.split(raw_data, raw_label)):
            print('k_fold', j, 'of', k_folds*k_fold_reps)

            train_data, train_labels = raw_data[train_index], raw_label[train_index]
            test_data, test_labels = raw_data[test_index], raw_label[test_index]

            train_labels = to_categorical(train_labels, num_classes=num_classes)
            test_labels = to_categorical(test_labels, num_classes=num_classes)

            valid_features = np.where(np.abs(train_data).sum(axis=0) > 0)[0]
            if len(valid_features) < train_data.shape[1]:
                print('Removing', train_data.shape[1] - len(valid_features), 'zero features')
                train_data = train_data[:, valid_features]
                test_data = test_data[:, valid_features]

            model_kwargs = {
                'mu': mu / len(train_data),
                'kernel': kernel,
                'degree': 3
            }
            print('mu :', model_kwargs['mu'], ', batch_size :', batch_size)

            svc_kwargs = {
                'C': 1.0,
                'solver': 0.
            }

            for i, n_features in enumerate([5, 10, 15, 20]):
                n_accuracies = []
                n_svc_accuracies = []
                n_model_accuracies = []
                n_BAs = []
                n_svc_BAs = []
                n_model_BAs = []
                n_mAPs = []
                n_svc_mAPs = []
                n_model_mAPs = []
                n_train_accuracies = []
                print('n_features : ', n_features)

                heatmaps = []
                for r in range(reps):
                    model = train_Keras(
                        train_data, train_labels, test_data, test_labels, model_kwargs,
                        e2efs_class=e2efs_class, n_features=n_features, e2efs_kwargs=e2efs_kwargs
                    )
                    heatmaps.append(K.eval(model.heatmap))
                    test_data_norm = model.normalization.transform(test_data)
                    train_data_norm = model.normalization.transform(train_data)
                    test_pred = model.predict(test_data_norm)
                    n_model_accuracies.append(model.evaluate(test_data_norm, test_labels, verbose=0)[-1])
                    n_model_BAs.append(balance_accuracy(test_labels, test_pred))
                    n_model_mAPs.append(average_precision_score(test_labels[:, -1], test_pred))
                    train_acc = model.evaluate(train_data_norm, train_labels, verbose=0)[-1]
                    print('n_features : ', n_features,
                          ', accuracy : ', n_model_accuracies[-1],
                          ', BA : ', n_model_BAs[-1],
                          ', mAP : ', n_model_mAPs[-1],
                          ', train_accuracy : ', train_acc)
                    del model
                    K.clear_session()

                heatmap = np.mean(heatmaps, axis=0)
                best_features = np.argsort(heatmap)[::-1][:n_features]

                svc_train_data = train_data[:, best_features]
                svc_test_data = test_data[:, best_features]

                norm = normalization_func()
                svc_train_data_norm = norm.fit_transform(svc_train_data)
                svc_test_data_norm = norm.transform(svc_test_data)

                bestcv = -1
                bestc = None
                bestSolver = None
                for s in [0, 1, 2, 3]:
                    for my_c in [0.001, 0.1, 0.5, 1.0, 1.4, 1.5, 1.6, 2.0, 2.5, 5.0, 100.0]:
                        cmd = '-v 5 -s ' + str(s) + ' -c ' + str(my_c) + ' -q'
                        cv = liblinearutil.train((2 * train_labels[:, -1] - 1).tolist(), svc_train_data_norm.tolist(), cmd)
                        if cv > bestcv:
                            # print('Best -> C:', my_c, ', s:', s, ', acc:', cv)
                            bestcv = cv
                            bestc = my_c
                            bestSolver = s
                svc_kwargs['C'] = bestc
                svc_kwargs['solver'] = bestSolver
                print('Best -> C:', bestc, ', s:', bestSolver, ', acc:', bestcv)

                for r in range(reps):
                    model = train_SVC(svc_train_data_norm, train_labels, svc_kwargs)
                    _, accuracy, test_pred = liblinearutil.predict(
                        (2 * test_labels[:, -1] - 1).tolist(), svc_test_data_norm.tolist(), model, '-q'
                    )
                    test_pred = np.asarray(test_pred)
                    n_svc_accuracies.append(accuracy[0])
                    n_svc_BAs.append(balance_accuracy(test_labels, test_pred))
                    n_svc_mAPs.append(average_precision_score(test_labels[:, -1], test_pred))
                    del model
                    model = train_Keras(svc_train_data, train_labels, svc_test_data, test_labels, model_kwargs)
                    train_data_norm = model.normalization.transform(svc_train_data)
                    test_data_norm = model.normalization.transform(svc_test_data)
                    test_pred = model.predict(test_data_norm)
                    n_BAs.append(balance_accuracy(test_labels, test_pred))
                    n_mAPs.append(average_precision_score(test_labels[:, -1], test_pred))
                    n_accuracies.append(model.evaluate(test_data_norm, test_labels, verbose=0)[-1])
                    n_train_accuracies.append(model.evaluate(train_data_norm, train_labels, verbose=0)[-1])
                    del model
                    K.clear_session()
                    print(
                        'n_features : ', n_features,
                        ', acc : ', n_accuracies[-1],
                        ', BA : ', n_BAs[-1],
                        ', mAP : ', n_mAPs[-1],
                        ', train_acc : ', n_train_accuracies[-1],
                        ', svc_acc : ', n_svc_accuracies[-1],
                        ', svc_BA : ', n_svc_BAs[-1],
                        ', svc_mAP : ', n_svc_mAPs[-1],
                    )
                if i >= len(accuracies):
                    accuracies.append(n_accuracies)
                    svc_accuracies.append(n_svc_accuracies)
                    model_accuracies.append(n_model_accuracies)
                    BAs.append(n_BAs)
                    mAPs.append(n_mAPs)
                    svc_BAs.append(n_svc_BAs)
                    svc_mAPs.append(n_svc_mAPs)
                    model_BAs.append(n_model_BAs)
                    model_mAPs.append(n_model_mAPs)
                    nfeats.append(n_features)
                    mus.append(model_kwargs['mu'])
                else:
                    accuracies[i] += n_accuracies
                    svc_accuracies[i] += n_svc_accuracies
                    model_accuracies[i] += n_model_accuracies
                    BAs[i] += n_BAs
                    mAPs[i] += n_mAPs
                    svc_BAs[i] += n_svc_BAs
                    svc_mAPs[i] += n_svc_mAPs
                    model_BAs[i] += n_model_BAs
                    model_mAPs[i] += n_model_mAPs

        output_filename = directory + 'LinearSVC_' + kernel + '_' + e2efs_class.__name__ + '.json'

        if not os.path.isdir(directory):
            os.makedirs(directory)

        info_data = {
            'kernel': kernel,
            'reps': reps,
            'classification': {
                'mus': mus,
                'n_features': nfeats,
                'accuracy': accuracies,
                'mean_accuracy': np.array(accuracies).mean(axis=1).tolist(),
                'svc_accuracy': svc_accuracies,
                'mean_svc_accuracy': np.array(svc_accuracies).mean(axis=1).tolist(),
                'model_accuracy': model_accuracies,
                'mean_model_accuracy': np.array(model_accuracies).mean(axis=1).tolist(),
                'BA': BAs,
                'mean_BA': np.array(BAs).mean(axis=1).tolist(),
                'mAP': mAPs,
                'mean_mAP': np.array(mAPs).mean(axis=1).tolist(),
                'svc_BA': svc_BAs,
                'svc_mean_BA': np.array(svc_BAs).mean(axis=1).tolist(),
                'svc_mAP': svc_mAPs,
                'svc_mean_mAP': np.array(svc_mAPs).mean(axis=1).tolist(),
                'model_BA': model_BAs,
                'model_mean_BA': np.array(model_BAs).mean(axis=1).tolist(),
                'model_mAP': model_mAPs,
                'model_mean_mAP': np.array(model_mAPs).mean(axis=1).tolist(),
            }
        }

        for k, v in info_data['classification'].items():
            if 'mean' in k:
                print(k, v)

        with open(output_filename, 'w') as outfile:
            json.dump(info_data, outfile)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)) + '/../../../')
    main(dataset_name)
