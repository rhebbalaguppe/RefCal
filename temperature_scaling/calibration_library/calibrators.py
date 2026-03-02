import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.std import tqdm

from utils import EarlyStopping, AverageMeter
from transformers.modeling_outputs import SequenceClassifierOutput

def _freeze_model(model : nn.Module):
    for params in model.parameters():
        params.requires_grad = False

class TemperatureScaling(nn.Module):
    def __init__(self, base_model, freeze=1, T=1.0):
        super().__init__()
        self.base_model = base_model
        self.T = T
        if freeze:
            _freeze_model(self.base_model)
            self.base_model.eval()

        # set up temperature list
        self.temp_list = []
        t = 0.2
        init_t = 0.2
        while init_t <= 10:
            self.temp_list.append(init_t)
            init_t += t

    def forward(self, x):
        #x = self.base_model(x)
        #x = x['input_ids'].squeeze(1).cpu()
        x = self.base_model(x)
        if isinstance(x, SequenceClassifierOutput):
           x = x.logits
        x = x.cuda()
        x /= self.T
        return x
    
    def calibrate(self, train_loader, args, **kwargs):

        min_error = float('inf')
        min_T = 1.0

        criterion = nn.CrossEntropyLoss()

        for T in tqdm(self.temp_list, desc="Running temp scaling"):
            error = 0.
            for images, targets in train_loader:
                #images, targets = images.cuda(), targets.cuda()
                if "bert" in args.model or "bert" in args.teacher:
                    images = images['input_ids'].squeeze(1)
                    outputs = self.base_model(images).logits
                else:
                    outputs = self.base_model(images)
                    
                outputs /= T

                cur_error = criterion(outputs, targets)
                error += cur_error.item()
            
            if error < min_error:
                min_T = T
                min_error = error

        self.T = min_T
        print(f"Best Temperature for Calibration: {self.T}")

class DirichletScaling(nn.Module):
    def __init__(self, base_model, num_classes, optim='adam', Lambda=0., Mu=0.):
        super().__init__()

        self.base_model = base_model
        self.num_classes = num_classes

        self.optim = optim
        self.Lambda = Lambda
        self.Mu = Mu

        _freeze_model(self.base_model)
        self.setup_model()

    def setup_model(self):
        self.fc = nn.Linear(self.num_classes, self.num_classes)
    
    def forward(self, x):
        x = self.base_model(x)
        x = torch.log_softmax(x, dim=1)
        x = self.fc(x)
        return x

    def regularizer(self):
        k = self.num_classes
        W, b = self.fc.parameters()

        # keep loss value 
        w_loss = ((W**2).sum() - (torch.diagonal(W, 0)**2).sum())/(k*(k-1))
        b_loss = ((b**2).sum())/k

        return self.Lambda*w_loss + self.Mu*b_loss

    def loss_func(self, outputs, targets):
        crit = nn.CrossEntropyLoss()
        return crit(outputs, targets) + self.regularizer()

    def give_params(self):
        return self.fc.parameters()

    def fit(self, train_loader, lr=0.001, epochs=25, patience=10):

        self.train()

        # if self.optim == "sgd":
        #     optimizer = optim.SGD(self.give_params(), 
        #                         lr=lr,
        #                         weight_decay=0.0)

        # elif self.optim == "adam":
        optimizer = optim.Adam(self.give_params(),
                            lr=lr,
                            weight_decay=0.0)
        
        scheduler = EarlyStopping(patience=patience)

        # send model to gpu
        self.cuda()

        last_loss = 0.0

        bar = tqdm(range(epochs), desc="running dir for ({:.2f},{:.2f})".format(self.Lambda, self.Mu))
        for i in bar:
        # for i in range(epochs):
            avg_loss = AverageMeter()
            for imgs, labels in train_loader:
                optimizer.zero_grad()
                imgs, labels = imgs.cuda(), labels.cuda()

                outs = self.forward(imgs)
                loss = self.loss_func(outs, labels)

                loss.backward()
                optimizer.step()

                avg_loss.update(loss.item())
            
            last_loss = avg_loss.avg
            bar.set_postfix_str("loss : {:.5f} | lr : {:.5f}".format(avg_loss.avg, lr))
            if scheduler.step(avg_loss.avg):
                break
        
        return last_loss
    
    def calibrate(self, train_loader, lr=0.001, epochs=25, double_fit=True, patience=10):

        loss = self.fit(train_loader, lr, epochs, patience)

        if double_fit:
            print("Trying to double fit...")
            lr /= 10
            loss = self.fit(train_loader, lr, epochs, patience)
        
        return loss


