import torch
import os


class Trainer:
    def __init__(self, criterion, optimizer, model, device):
        self.criterion = criterion
        self.optimizer = optimizer
        self.model = model
        self.device = device

    def loop(self, num_epochs, train_loader, val_loader):
        for epoch in range(1, num_epochs + 1):
            self.train(train_loader, epoch, num_epochs)
            self.val(val_loader, epoch, num_epochs)

    def train(self, dataloader, epoch, num_epochs):
        self.model.train()
        with torch.enable_grad():
            self._iteration_train(dataloader, epoch, num_epochs)

    def val(self, dataloader, epoch, num_epochs):
        self.model.eval()
        with torch.no_grad():
            self._iteration_val(dataloader)

    def test(self, data_loader):
        self.model.eval()
        correct, total = 0, 0
        for batch in data_loader:
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            test_acc = correct / total

            # print('Accuracy on the train set: %f %%' % (100 * train_acc))
            print('Accuracy on the test set: %f %%' % (100 * test_acc))

    def _iteration_train(self, dataloader, epoch, num_epochs):
        total_step = len(dataloader)
        tot_loss = 0.0
        correct = 0
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            _, preds = torch.max(outputs.data, 1)
            loss = self.criterion(outputs, labels)

            # Backward and optimizer
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            tot_loss += loss.data
            if (i + 1) % 2 == 0:
                print('Epoch: [{}/{}], Step: [{}/{}], Loss: {:.4f}'
                      .format(epoch, num_epochs, i + 1, total_step, loss.item()))
            correct += torch.sum(preds == labels.data).to(torch.float32)
        ### Epoch info ####
        epoch_loss = tot_loss / len(dataloader.dataset)
        print('train loss: {:.4f}'.format(epoch_loss))
        epoch_acc = correct / len(dataloader.dataset)
        print('train acc: {:.4f}'.format(epoch_acc))
        if epoch % num_epochs == 0:
            # state = {
            #     'model': self.model.state_dict(),
            #     'optimizer': self.optimizer.state_dict(),
            #     'epoch': epoch,
            #     'train_loss': epoch_loss,
            #     'train_acc': epoch_acc,
            # }
            save_path = "output"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(self.model.state_dict(), save_path + "/" + "epoch" + str(epoch) + "-resnet121" + ".t7")

    def _iteration_val(self, dataloader):
        total_step = len(dataloader)
        tot_loss = 0.0
        correct = 0
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            _, preds = torch.max(outputs.data, 1)
            loss = self.criterion(outputs, labels)
            tot_loss += loss.data
            correct += torch.sum(preds == labels.data).to(torch.float32)
            if (i + 1) % 2 == 0:
                print('Epoch: [{}/{}], Step: [{}/{}], Loss: {:.4f}'
                      .format(1, 1, i + 1, total_step, loss.item()))
        ### Epoch info ####
        epoch_loss = tot_loss / len(dataloader.dataset)
        print('val loss: {:.4f}'.format(epoch_loss))
        epoch_acc = correct / len(dataloader.dataset)
        print('val acc: {:.4f}'.format(epoch_acc))
