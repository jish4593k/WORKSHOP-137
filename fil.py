import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load iris dataset
iris = sns.load_dataset("iris")

# Separate features and targets
features = iris.iloc[:, [0, 2]].values
targets = iris['species']

# Convert targets to numerical labels
class_mapping = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
numeric_targets = targets.map(class_mapping).values

# Convert to PyTorch tensors
features_tensor = torch.tensor(features, dtype=torch.float32)
targets_tensor = torch.tensor(numeric_targets, dtype=torch.long)
s
indices = torch.randperm(len(features_tensor))
train_size = int(0.8 * len(features_tensor))
train_indices, test_indices = indices[:train_size], indices[train_size:]

features_train, features_test = features_tensor[train_indices], features_tensor[test_indices]
targets_train, targets_test = targets_tensor[train_indices], targets_tensor[test_indices]

mean = features_train.mean(dim=0)
std = features_train.std(dim=0)
features_train = (features_train - mean) / std
features_test = (features_test - mean) / std

class SVM(nn.Module):
    def __init__(self):
        super(SVM, self).__init__()
        self.linear = nn.Linear(2, 3)

    def forward(self, x):
        return self.linear(x)

svm = SVM()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(svm.parameters(), lr=0.01)


num_epochs = 1000
for epoch in range(num_epochs):
    outputs = svm(features_train)
    loss = criterion(outputs, targets_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


svm.eval()
with torch.no_grad():
    predictions = torch.argmax(svm(features_test), dim=1)

num_correct = torch.sum(predictions == targets_test).item()
num_testing = len(features_test)
accuracy = num_correct / num_testing * 100

print("No. correct={}, No. testing examples={}, prediction accuracy={} per cent".format(
    num_correct, num_testing, round(accuracy, 2)))


def plot_decision_regions_tensor(features, targets, model, title):
    xx, yy = torch.meshgrid(torch.arange(features[:, 0].min(), features[:, 0].max(), 0.01),
                            torch.arange(features[:, 1].min(), features[:, 1].max(), 0.01))
    input_tensor = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1)
    with torch.no_grad():
        predictions = torch.argmax(model(input_tensor), dim=1).reshape(xx.shape)
    plt.contourf(xx.numpy(), yy.numpy(), predictions.numpy(), alpha=0.8)
    plt.scatter(features[:, 0], features[:, 1], c=targets, cmap='viridis', edgecolors='k')
    plt.xlabel('sepal length [CM]')
    plt.ylabel('petal length [CM]')
    plt.title(title)
    plt.show()

plot_decision_regions_tensor(features_train, targets_train, svm, 'SVM on Iris')
