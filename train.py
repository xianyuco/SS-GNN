import os
import time
import copy

import torch
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from lifelines.utils import concordance_index

from model import GINNet as DeepMice
from dataset import get_gin_dataloader
from utils import set_data_device

import warnings
warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Trainer:
    '''
            Drug Target Binding Affinity
    '''

    def __init__(self, device='cuda:0'):
        self.lr = 0.001
        self.decay = 0.00001
        self.BATCH_SIZE = 64
        self.train_epoch = 1000

        self.path = '../data/mice_features2/all'
        self.label_path = '../data/mice_features2/total_labels.pkl'

        self.device = device
        self.config = None

        self.model = DeepMice()
        self.model = self.model.to(device)
        self.trainloader = get_gin_dataloader(self.path,
                                              self.label_path,
                                              self.BATCH_SIZE, phase='train')
        self.testloader = get_gin_dataloader(self.path,
                                             self.label_path,
                                             self.BATCH_SIZE,
                                             phase='test')
        self.valloader = get_gin_dataloader(self.path,
                                            self.label_path,
                                            self.BATCH_SIZE,
                                            phase='val')
        self.loss_fct = torch.nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(),
                                    lr=self.lr)

    def test_(self):
        y_pred = []
        y_label = []
        self.model.eval()
        for i, (x, y) in enumerate(self.testloader):
            x, y = set_data_device(
                (x, y), self.device)
            score = self.model(x)
            logits = torch.squeeze(score).detach().cpu().numpy()
            label_ids = y.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()
        self.model.train()
        return [mean_squared_error(y_label, y_pred),
                pearsonr(y_label, y_pred)[0],
                pearsonr(y_label, y_pred)[1],
                concordance_index(y_label, y_pred), y_pred]

    def val_(self):
        y_pred = []
        y_label = []
        self.model.eval()
        for i, (x, y) in enumerate(self.valloader):
            x, y = set_data_device(
                (x, y), self.device)
            score = self.model(x)
            logits = torch.squeeze(score).detach().cpu().numpy()
            label_ids = y.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()
        self.model.train()
        return [mean_squared_error(y_label, y_pred),
                pearsonr(y_label, y_pred)[0],
                pearsonr(y_label, y_pred)[1],
                concordance_index(y_label, y_pred), y_pred]

    def train(self):
        max_MSE = 10000
        model_max = copy.deepcopy(self.model)
        print('--- Go for Training ---')
        t_start = time.time()
        currentPearson = 0.0
        for epo in range(self.train_epoch):
            for i, (x, y) in enumerate(self.trainloader):
                x, y = set_data_device(
                    (x, y), self.device)
                score = self.model(x)
                loss = self.loss_fct(score.squeeze(1), y)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if (i % 100 == 0):
                    t_now = time.time()
                    print(
                        'Training at Epoch ' + str(epo + 1) +
                        ' iteration ' + str(i) +
                        ' with loss ' + str(loss.cpu().detach().numpy())[:7] +
                        ". Total time " + str(int(t_now - t_start)/3600)[:7] +
                        " hours")

            # validate, select the best model up to now
            with torch.set_grad_enabled(False):
                mse, r2, p_val, CI, logits = self.val_()
                if mse < max_MSE:
                    model_max = copy.deepcopy(self.model)
                    max_MSE = mse
                print('Validation at Epoch ' + str(epo + 1) +
                      ' , MSE: ' + str(mse)[:7] +
                      ' , Pearson Correlation: ' + str(r2)[:7] +
                      ' with p-value: ' + str(p_val)[:7] +
                      ' , Concordance Index: '+str(CI)[:7])
            self.save_model('saved_models/model_'+str(epo)+'.pt')
            if r2 > currentPearson:
                self.save_model('best_models/model_' + str(epo) + '.pt')
                currentPearson = r2
        self.model = model_max
        print('--- Go for Testing ---')
        mse, r2, p_val, CI, logits = self.test_()
        print('Testing MSE: ' + str(mse) +
              ' , Pearson Correlation: ' + str(r2)
              + ' with p-value: ' + str(p_val) +
              ' , Concordance Index: '+str(CI))
        print('--- Training Finished ---')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        # save_dict(path, self.config)

    def load_pretrained(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        if self.device == 'cuda':
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location=torch.device('cpu'))
        # to support training from multi-gpus data-parallel:

        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict
        self.model.load_state_dict(state_dict)


if __name__ == '__main__':
    trainer = Trainer()
    # trainer.load_pretrained(path="saved_models/model_364.pt")
    trainer.train()
    trainer.save_model('saved_models/model.pt')
