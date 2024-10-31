from genericpath import exists
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import models
import os
import argparse
from tqdm import tqdm
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO, Evaluator
from torch.utils import tensorboard

# Note that: here we provide a basic solution for training and testation.
# You can directly change it if you find something wrong or not good enough.

def run(model, train_set, valid_set, test_set, criterion, optimizer, scheduler, save_dir, data_path, num_epochs=20):

    writer = tensorboard.SummaryWriter(save_dir)
    
    def train(model, train_loader, optimizer, criterion):
        model.train(True)
        total_loss = 0.0
        bar = tqdm(train_loader, desc='train')
        for idx, (inputs, labels) in enumerate(bar):
            
            iter = epoch * len(train_loader) + idx + 1
            
            inputs = inputs.to(device) # [64, 3, 64, 64]
            labels = labels.to(device) # [64, 1]
            optimizer.zero_grad()
            outputs = model(inputs) # [64, 9]

            labels = labels.squeeze().long()
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            bar.set_postfix(loss=loss.item())
            
            # if iter % 100 == 0:
            #     writer.add_scalar('train_loss', loss.item(), iter)
            #     model.train(False)
            #     with torch.no_grad():
            #         val_auc, val_acc = valid_or_test(model, valid_loader, 'val')
            #         writer.add_scalar('val_auc', val_auc, iter)
            #         writer.add_scalar('val_acc', val_acc, iter)
            
        epoch_loss = total_loss / len(train_loader.dataset)
        return epoch_loss

    def valid_or_test(model, valid_loader, split):
        model.train(False)

        y_true = torch.tensor([])
        y_score = torch.tensor([])
        for inputs, labels in tqdm(valid_loader, desc=split):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            
            labels = labels.squeeze().long()
            outputs = outputs.softmax(dim=-1)
            labels = labels.float().resize_(len(labels), 1)


            y_true = torch.cat((y_true, labels.cpu()), 0)
            y_score = torch.cat((y_score, outputs.cpu()), 0)
        
        scheduler.step()

        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()
        
        evaluator = Evaluator(data_flag, split, size=64, root=data_path)
        metrics = evaluator.evaluate(y_score)
    
        
        return metrics[0], metrics[1]
    
    best_acc = 0.0
    if not exists(save_dir):
        os.makedirs(save_dir)
    for epoch in range(num_epochs):
        print('epoch:{:d}/{:d}'.format(epoch, num_epochs))
        print('*' * 100)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=8)
        
        train_loss = train(model, train_loader, optimizer, criterion)
        print("training: {:.4f}".format(train_loss))

        with torch.no_grad():
            val_auc, val_acc = valid_or_test(model, valid_loader, 'val')
        print('valid auc: %.3f  acc:%.3f' % (val_auc, val_acc))
        
        print()
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
    with torch.no_grad():
        test_auc, test_acc = valid_or_test(model, test_loader, 'test')
    print('test auc: %.3f  acc:%.3f' % (test_auc, test_acc))
        
    torch.save(best_model, os.path.join(save_dir, 'best_model.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hw1')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/', help='model save path')
    args = parser.parse_args()

    # data preparation

    data_flag = 'pathmnist'
    download = False
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels'] # 3
    n_classes = len(info['label']) # 9
    DataClass = getattr(medmnist, info['python_class'])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # download dataset first and modify the data_path accordingly
    data_path = './data'
    train_dataset = DataClass(root=data_path, split='train', transform=data_transform, size=64, download=download)
    valid_dataset = DataClass(root=data_path, split='val', transform=data_transform, size=64, download=download)
    test_dataset = DataClass(root=data_path, split='test', transform=data_transform, size=64, download=download)

    # about training
    num_epochs = 5
    lr = 0.001
    batch_size = 64

    # model initialization
    model = models.model_B(num_classes=n_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    run(model, train_dataset, valid_dataset, test_dataset, criterion, optimizer, scheduler, args.save_dir, data_path, num_epochs=num_epochs)
