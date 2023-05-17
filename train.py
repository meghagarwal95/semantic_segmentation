import torch.cuda
from torch.utils.data import DataLoader, random_split
from datetime import datetime
import torch
from ade20k import ADE20KDataset
from models import DeeplabV3

class Trainer:
    def __init__(self, root_folder, batch_size=2, num_workers=2, val_split=0.3, learning_rate=0.001, momentum=0.9, n_epochs=10, model_type='deeplabv3'):
        dataset = ADE20KDataset(root_folder)
        self.trainDataset, self.valDataset = random_split(dataset, [1-val_split, val_split])
        #self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.model_type = model_type
        self.num_classes = dataset.num_classes
        self.n_epochs = n_epochs
        self._init_load()
        self._load_optimizer()

    def _init_load(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Device used: ', self.device)
        self.train_dataloader = DataLoader(self.trainDataset, shuffle=True, batch_size=self.batch_size,
                                           num_workers=self.num_workers)
        self.val_dataloader = DataLoader(self.valDataset, shuffle=True, batch_size=self.batch_size,
                                         num_workers=self.num_workers)
        net = DeeplabV3(self.num_classes)
        self.model = net.model.to(self.device)

    def _load_optimizer(self):
        # SGD for optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        return

    def lossFunction(self, outputs, labels, ignore_index=255):
        if (len(labels.shape) == 4) and (labels.shape[1] == 1):
            labels = torch.squeeze(labels, 1)

        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        return loss_fn(outputs, labels)

    def _train_one_epoch(self, epoch_idx):
        """
        One train epoch definition
        :return:
        """
        running_loss = 0.

        for i, data in enumerate(self.train_dataloader):
            #print('DEBUGG: ', data[1].shape)
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # helper function to run predictions on a batch and get loss
            loss = self._run_predictions_on_a_batch(data)

            # Compute the gradients
            loss.backward()

            # Update weights and adjust learning rate
            self.optimizer.step()

            # gather loss and report
            running_loss += loss.item()
            print(loss.item())
            if i % 1000 == 999:
                last_loss = running_loss/1000
                print(' batch {} loss: {}'.format(i+1, last_loss))
                running_loss = 0.

        return last_loss

    def _run_predictions_on_a_batch(self, data):
        # Get next batch of data and labels
        input, label = data[0].to(self.device), data[1].to(self.device)

        # Make predictions for this batch
        output = self.model(input)['out']

        # Compute the loss
        loss = self.lossFunction(output, label.long())

        return loss

    def run(self):

        best_val_loss = 1_000_000.

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        for epoch_idx in range(self.n_epochs):
            print('EPOCH {}:'.format(epoch_idx+1))

            # Make sure gradient tracking is on, and do pass over the data
            self.model.train(True)

            avg_loss = self._train_one_epoch(epoch_idx)

            running_val_loss = 0.
            for i, val_data in enumerate(self.val_dataloader):
                val_loss = self._run_predictions_on_a_batch(val_data)
                running_val_loss += val_loss.item()

            avg_val_loss = running_val_loss/(i+1)
            print('LOSS train {} val {}'.format(avg_loss, avg_val_loss))

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_path = 'checkpoints/{}/model_{}_{}.pt'.format(self.model_type, timestamp, epoch_idx+1)
                torch.save(self.model.state_dict(), model_path)

        return