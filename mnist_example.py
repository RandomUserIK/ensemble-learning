import torch.nn as nn
from torch.nn import functional as f
from torch.utils import data as torch_data
from torchensemble import FusionClassifier, VotingClassifier, BaggingClassifier, GradientBoostingClassifier, \
    SnapshotEnsembleClassifier, AdversarialTrainingClassifier, FastGeometricClassifier, SoftGradientBoostingClassifier
from torchensemble.utils.logging import set_logger
from torchvision import transforms, datasets

logger = set_logger('classification_mnist_mlp')

train = datasets.MNIST('./dataset', train=True, download=True, transform=transforms.ToTensor())
test = datasets.MNIST('./dataset', train=False, transform=transforms.ToTensor())

train_loader = torch_data.DataLoader(train, batch_size=128, shuffle=True)
test_loader = torch_data.DataLoader(test, batch_size=128, shuffle=True)

optimizers = ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD']


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(784, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 10)

    def forward(self, data):
        data = data.view(data.size(0), -1)  # flatten
        output = f.relu(self.linear1(data))
        output = f.relu(self.linear2(output))
        output = self.linear3(output)
        return output


def fusion_classifier():
    fusion_ensemble = FusionClassifier(
        estimator=MLP,
        n_estimators=10,
        cuda=True
    )
    fusion_ensemble.set_optimizer('Adam', lr=1e-3, weight_decay=5e-4)
    fusion_ensemble.fit(train_loader, epochs=5)
    return fusion_ensemble.evaluate(test_loader=test_loader)


def voting_classifier():
    voting_ensemble = VotingClassifier(
        estimator=MLP,
        n_estimators=10,
        cuda=True
    )
    voting_ensemble.set_optimizer('Adam', lr=1e-3, weight_decay=5e-4)
    voting_ensemble.fit(train_loader, epochs=5)
    return voting_ensemble.evaluate(test_loader=test_loader)


def bagging_classifier():
    bagging_ensemble = BaggingClassifier(
        estimator=MLP,
        n_estimators=10,
        cuda=True
    )
    bagging_ensemble.set_optimizer('Adam', lr=1e-3, weight_decay=5e-4)
    bagging_ensemble.fit(train_loader, epochs=5)
    return bagging_ensemble.evaluate(test_loader=test_loader)


def gradient_boosting_classifier():
    gradient_boosting_ensemble = GradientBoostingClassifier(
        estimator=MLP,
        n_estimators=10,
        cuda=True
    )
    gradient_boosting_ensemble.set_optimizer('Adam', lr=1e-3, weight_decay=5e-4)
    gradient_boosting_ensemble.fit(train_loader, epochs=5)
    return gradient_boosting_ensemble.evaluate(test_loader=test_loader)


def snapshot_classifier():
    snapshot_ensemble = SnapshotEnsembleClassifier(
        estimator=MLP,
        n_estimators=10,
        cuda=True
    )
    snapshot_ensemble.set_optimizer('Adam', lr=1e-3, weight_decay=5e-4)
    # The number of training epochs should be a multiple of n_estimators.
    snapshot_ensemble.fit(train_loader, epochs=10)
    return snapshot_ensemble.evaluate(test_loader=test_loader)


def adversarial_classifier():
    adversarial_ensemble = AdversarialTrainingClassifier(
        estimator=MLP,
        n_estimators=10,
        cuda=True
    )
    adversarial_ensemble.set_optimizer('Adam', lr=1e-3, weight_decay=5e-4)
    adversarial_ensemble.fit(train_loader, epochs=5, epsilon=0.01)
    return adversarial_ensemble.evaluate(test_loader=test_loader)


def fast_geometric_classifier():
    fast_geometric_ensemble = FastGeometricClassifier(
        estimator=MLP,
        n_estimators=10,
        cuda=True
    )
    fast_geometric_ensemble.set_optimizer('Adam', lr=1e-3, weight_decay=5e-4)
    fast_geometric_ensemble.fit(train_loader, epochs=5)
    return fast_geometric_ensemble.evaluate(test_loader=test_loader)


def soft_gradient_boosting_classifier():
    soft_gradient_boosting_ensemble = SoftGradientBoostingClassifier(
        estimator=MLP,
        n_estimators=10,
        cuda=True
    )
    soft_gradient_boosting_ensemble.set_optimizer('Adam', lr=1e-3, weight_decay=5e-4)
    soft_gradient_boosting_ensemble.fit(train_loader, epochs=5)
    return soft_gradient_boosting_ensemble.evaluate(test_loader=test_loader)


if __name__ == '__main__':
    fusion_ensemble_accuracy = fusion_classifier()
    voting_ensemble_accuracy = voting_classifier()
    bagging_ensemble_accuracy = bagging_classifier()
    gradient_boosting_ensemble_accuracy = gradient_boosting_classifier()
    snapshot_ensemble_accuracy = snapshot_classifier()
    adversarial_ensemble_accuracy = adversarial_classifier()
    fast_geometric_ensemble_accuracy = fast_geometric_classifier()
    soft_gradient_boosting_ensemble_accuracy = soft_gradient_boosting_classifier()

    print('-' * 35)
    print(f'Fusion ensemble accuracy: {fusion_ensemble_accuracy}%')
    print(f'Voting ensemble accuracy: {voting_ensemble_accuracy}%')
    print(f'Bagging ensemble accuracy: {bagging_ensemble_accuracy}%')
    print(f'Gradient Boosting ensemble accuracy: {gradient_boosting_ensemble_accuracy}%')
    print(f'Snapshot ensemble accuracy: {snapshot_ensemble_accuracy}%')
    print(f'Adversarial ensemble accuracy: {adversarial_ensemble_accuracy}%')
    print(f'Fast Geometric ensemble accuracy: {fast_geometric_ensemble_accuracy}%')
    print(f'Soft Gradient Boosting ensemble accuracy: {soft_gradient_boosting_ensemble_accuracy}%')
