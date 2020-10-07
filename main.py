from __future__ import print_function
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from models.LeNet import LeNet
from models.AlexNet import AlexNet
from models.CNN import cnn
from models.ResNet import ResNet
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler  # for validation set
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import numpy as np
import matplotlib.pyplot as plt


# functions to show an image
def imsave(img):
    npimg = img.numpy()
    npimg = (np.transpose(npimg, (1, 2, 0)) * 255).astype(np.uint8)
    im = Image.fromarray(npimg)
    im.save("./results/your_file.jpeg")


criterion = nn.CrossEntropyLoss()


# used for confusion matrix
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


# training function for all 4 models

def train_cnn(log_interval, model, device, train_loader, optimizer, epoch, train_loss, train_acc):
    model.train()
    correct = 0
    trainloss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        output = model(data)

        loss = criterion(output, target)
        trainloss += loss
        # train_loss.append(loss)

        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    trainloss /= len(train_loader.dataset)
    train_loss.append(trainloss)
    train_acc.append(100. * correct / len(train_loader.dataset))


def validation_cnn(log_interval, model, device, valid_loader, optimizer, epoch, Val_loss):
    model.train()
    valloss = 0
    for batch_idx, (data, target) in enumerate(valid_loader):
        data, target = data.to(device), target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimizer
        output = model(data)

        loss = criterion(output, target)
        # Val_loss.append(loss)
        valloss += loss
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Valid Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(valid_loader.dataset),
                       100. * batch_idx / len(valid_loader), loss.item()))
    valloss /= len(valid_loader.dataset)
    Val_loss.append(valloss)


def test(model, device, test_loader, test_acc, testLoss):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    testLoss.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_acc.append(100. * correct / len(test_loader.dataset))


def main():
    epoches = 1
    gamma = 0.1
    log_interval = 10
    torch.manual_seed(1)
    save_model = True

    LeN = True
    AleN = False
    CNN = False
    ResN = False

    # Check whether you can use Cuda
    use_cuda = torch.cuda.is_available()
    # Use Cuda if you can
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Load data for training and test sets
    trainset = datasets.FashionMNIST('../data/FashionMNIST', download=True, train=True, transform=transforms.Compose([
        transforms.ToTensor()
    ]))
    testset = datasets.FashionMNIST('../data/FashionMNIST', download=True, train=False, transform=transforms.Compose([
        transforms.ToTensor()
    ]))

    # gets 20% of the train set
    indices = list(range(len(trainset)))
    np.random.shuffle(indices)
    # to get 20% of the train set
    split = int(np.floor(0.2 * len(trainset)))
    train_sample = SubsetRandomSampler(indices[:split])
    valid_sample = SubsetRandomSampler(indices[:split])

    train_loader = torch.utils.data.DataLoader(trainset, sampler=train_sample, batch_size=64, **kwargs)
    valid_loader = torch.utils.data.DataLoader(trainset, sampler=valid_sample, batch_size=64, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, **kwargs)

    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    img = torchvision.utils.make_grid(images)
    imsave(img)

    # Build and run network
    if LeN:
        model = LeNet().to(device)
    elif AleN:
        model = AlexNet().to(device)
    elif CNN:
        model = cnn().to(device)
    else:
        model = ResNet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    scheduler = StepLR(optimizer, step_size=2, gamma=gamma)

    train_loss = []
    Val_Loss = []
    test_acc = []
    train_acc = []
    testLoss = []

    since = time.time()

    for epoch in range(1, epoches + 1):
        validation_cnn(log_interval, model, device, valid_loader, optimizer, epoch, Val_Loss)
        train_cnn(log_interval, model, device, train_loader, optimizer, epoch, train_loss, train_acc)

        test(model, device, test_loader, test_acc, testLoss)
        scheduler.step()

    bestTest_acc = 0.0
    for i in range(len(test_acc)):
        if test_acc[i] > bestTest_acc:
            bestTest_acc = test_acc[i]

    if save_model:
        if LeN:
            torch.save(model.state_dict(), "./results/fmnist_lenet.pt")
        elif AleN:
            torch.save(model.state_dict(), "./results/fmnist_alexent.pt")
        elif CNN:
            torch.save(model.state_dict(), "./results/fmnist_cnn.pt")
        else:
            torch.save(model.state_dict(), "./results/fmnist_resnet.pt")

    with torch.no_grad():  # to prevent leaking test data into the model
        def get_all_preds(model, loader):
            all_preds = torch.tensor([])
            for batch in loader:
                images, labels = batch

                preds = model(images)
                all_preds = torch.cat(
                    (all_preds, preds)
                    , dim=0
                )
            return all_preds

        prediction_loader = torch.utils.data.DataLoader(trainset, batch_size=10000)
        train_preds = get_all_preds(model, prediction_loader)
        preds_correct = get_num_correct(train_preds, trainset.targets)
        print('total correct:', preds_correct)
        print('accuracy:', preds_correct / len(trainset))

        cm = confusion_matrix(trainset.targets, train_preds.argmax(dim=1))

        # plots the confusion matrix as a graph
        def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
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
                plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')

        names = (
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

        # prints out precision,recall, f1 score
        print(classification_report(trainset.targets, train_preds.argmax(dim=1), target_names=names))

        plt.figure(figsize=(10, 10))
        plot_confusion_matrix(cm, names)
        plt.show()

    # training time and test accuracy

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Test Accuracy: {:.2f} %'.format(bestTest_acc))


if __name__ == '__main__':
    main()
