'''
Created: 2022-03-24 17:04:05
Author : Ziyi
Email : z1chen@whu.edu.cn
-----
Description: Graduation Design
'''

from utils.logger import Logger
import time
from torch.utils.data import DataLoader

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score

log = Logger().getLogger("__ocsvm__")

class OCSVM(object):
    def __init__(self, params, encoder):
        """Init OCSVM instance."""
        self.params = params
        self.encoder = encoder
        self.model = None
        self.linear_model = None
        self.results = {
            'train_time': None,
            'test_time': None,
            'test_auc': None,
            'test_scores': None,
            'train_time_linear': None,
            'test_time_linear': None,
            'test_auc_linear': None
        }

    def train(self, dataset):
        gammas = np.logspace(-7, 2, num=10, base=2)
        best_auc = 0.0
        train_loader = DataLoader(dataset=dataset.train_set, batch_size=128, shuffle=True,
                                  num_workers=self.params['num_of_workers'], drop_last=False)

        X = ()
        for data in train_loader:
            inputs, _, _, _ = data
            inputs = self.encoder(inputs.to(self.params['device']))
            X_batch = inputs.view(inputs.size(0), -1)
            X += (X_batch.cpu().data.numpy(),)
        X = np.concatenate(X)

        # Training
        log.info('Starting training...')

        # Sample hold-out set from test set
        _, test_loader = dataset.loaders(batch_size=128, num_workers=self.params['num_of_workers'])

        X_test = ()
        labels = []
        for data in test_loader:
            inputs, label_batch, _, _ = data
            inputs, label_batch = inputs.to(self.params['device']), label_batch.to(self.params['device'])
            inputs = self.encoder(inputs.to(self.params['device']))
            X_batch = inputs.view(inputs.size(0), -1)
            X_test += (X_batch.cpu().data.numpy(),)
            labels += label_batch.cpu().data.numpy().astype(np.int64).tolist()
        X_test, labels = np.concatenate(X_test), np.array(labels)
        n_test, n_normal, n_outlier = len(X_test), np.sum(labels == 0), np.sum(labels == 1)
        n_val = int(0.1 * n_test)
        n_val_normal, n_val_outlier = int(n_val * (n_normal/n_test)), int(n_val * (n_outlier/n_test))
        perm = np.random.permutation(n_test)
        X_val = np.concatenate((X_test[perm][labels[perm] == 0][:n_val_normal],
                                X_test[perm][labels[perm] == 1][:n_val_outlier]))
        labels = np.array([0] * n_val_normal + [1] * n_val_outlier)

        i = 1
        for gamma in gammas:
            # Model candidate
            model = OneClassSVM(kernel=self.params['kernel'], nu=self.params['nu'], gamma=gamma)

            start_time = time.time()
            model.fit(X)
            train_time = time.time() - start_time

            scores = (-1.0) * model.decision_function(X_val)
            scores = scores.flatten()

            auc = roc_auc_score(labels, scores)

            log.info(f'| Model {i:02}/{len(gammas):02} | Gamma: {gamma:.8f} | Train Time: {train_time:.3f}s '
                        f'| Val AUC: {100. * auc:.2f} |')

            if auc > best_auc:
                best_auc = auc
                self.model = model
                self.params['gamma'] = gamma
                self.results['train_time'] = train_time

            i += 1

        # linear model
        self.linear_model = OneClassSVM(kernel='linear', nu=self.nu)
        start_time = time.time()
        self.linear_model.fit(X)
        train_time = time.time() - start_time
        self.results['train_time_linear'] = train_time

        log.info(f'Best Model: | Gamma: {self.gamma:.8f} | AUC: {100. * best_auc:.2f}')
        log.info('Training Time: {:.3f}s'.format(self.results['train_time']))
        log.info('Finished training.')

    def test(self, dataset):
        _, test_loader = dataset.loaders(batch_size=128, num_workers=self.params['num_of_workers'])

        # Get data from loader
        idx_label_score = []
        X = ()
        idxs = []
        labels = []
        for data in test_loader:
            inputs, label_batch, _, idx = data
            inputs, label_batch, idx = inputs.to(self.params['device']), label_batch.to(self.params['device']), idx.to(self.params['device'])
            inputs = self.encoder(inputs)
            X_batch = inputs.view(inputs.size(0), -1)
            X += (X_batch.cpu().data.numpy(),)
            idxs += idx.cpu().data.numpy().astype(np.int64).tolist()
            labels += label_batch.cpu().data.numpy().astype(np.int64).tolist()
        X = np.concatenate(X)

        # Testing
        log.info('Starting testing...')
        start_time = time.time()

        scores = (-1.0) * self.model.decision_function(X)

        self.results['test_time'] = time.time() - start_time
        scores = scores.flatten()
        self.params['rho'] = -self.model.intercept_[0]

        # Save triples of (idx, label, score) in a list
        idx_label_score += list(zip(idxs, labels, scores.tolist()))
        self.results['test_scores'] = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.results['test_auc'] = roc_auc_score(labels, scores)

        # linear model
        start_time = time.time()
        scores_linear = (-1.0) * self.linear_model.decision_function(X)
        self.results['test_time_linear'] = time.time() - start_time
        scores_linear = scores_linear.flatten()
        self.results['test_auc_linear'] = roc_auc_score(labels, scores_linear)
        log.info('Test AUC linear model: {:.2f}%'.format(100. * self.results['test_auc_linear']))
        log.info('Test Time linear model: {:.3f}s'.format(self.results['test_time_linear']))

        # Log results
        log.info('Test AUC: {:.2f}%'.format(100. * self.results['test_auc']))
        log.info('Test Time: {:.3f}s'.format(self.results['test_time']))
        log.info('Finished testing.')