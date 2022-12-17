import torch
import torch.nn.functional as F
from torch import nn


class ROCKET:
    """Implementation of https://arxiv.org/abs/1910.13051"""
    
    def __init__(self, input_len:int, n_kernels:int=10000, kernels_sizes_set:torch.Tensor=torch.tensor([7, 9, 11]),
                 device:str='cpu', kernels_configurations:dict=None):
        self.input_len = input_len
        self.n_kernels = n_kernels
        self.kernels_sizes_set = kernels_sizes_set
        self.device = device
        if kernels_configurations is None:
            self.create_kernel_configurations()
        else:
            self.kernels_configurations = kernels_configurations
        
    def create_kernel_configurations(self):
        kernels_sizes = self.kernels_sizes_set[torch.randint(0, 3, size=(self.n_kernels, ))]        
        self.kernels_configurations = {
            i: dict(
                weight = torch.empty(1, 1, kernels_sizes[i], dtype=torch.double).normal_(0, 1).to(self.device),
                bias = torch.empty(1, dtype=torch.double).uniform_(-1, 1).to(self.device),
                padding = 'same' if (torch.rand(1).item() > 0.5) else 'valid',
                dilation = 2 ** torch.empty(1).uniform_(0, torch.log2((self.input_len - 1) / (kernels_sizes[i] - 1))).int().item()
            )
            for i in range(self.n_kernels)
        }
    
    @staticmethod
    def get_random_features(x:torch.Tensor, weight:torch.Tensor, bias: torch.Tensor, padding: str, dilation: int):
    
        x_convolved = F.conv1d(x,                             
                               weight=weight-weight.mean(),
                               bias=bias,
                               padding=padding,
                               dilation=dilation
                               )
        x_convolved_max = x_convolved.max(dim=2).values
        x_convolved_ppv = (x_convolved > 0).float().mean(dim=2)       
        return torch.cat([x_convolved_max, x_convolved_ppv], dim=1)
    
    def generate_random_features(self, x: torch.Tensor):
        features = []
        for i in range(self.n_kernels):
            features.append(self.get_random_features(x, **self.kernels_configurations[i]))
        return torch.cat(features, dim=1)
    
    
class LinearClassifier(nn.Module):
    def __init__(self, n_features, n_classes, p=0.):
        super(LinearClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(n_features, n_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)
    
    
class Logreg:
    
    def __init__(self, n_features, n_classes, p_dropout, lr, weight_decay, epochs_to_wait, tol, max_steps, ignore_first_n_steps, device):
        self.model = LinearClassifier(n_features, n_classes, p_dropout).double()
        self.model.to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3)
        self.loss = nn.CrossEntropyLoss()
        self.epochs_to_wait = epochs_to_wait
        self.tol = tol
        self.max_steps = max_steps
        self.ignore_first_n_steps = ignore_first_n_steps
    
    def train_step(self, x, y):
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        loss = self.loss(logits, y)
        loss.backward()
        self.optimizer.step()
        return loss.cpu().item()
        
    @torch.no_grad()
    def validate(self, x, y, mode='eval'):
        if mode == 'eval':
            self.model.eval()
        logits = self.model(x)        
        loss = self.loss(logits, y).cpu().item()
        acc = (torch.argmax(logits, dim=1) == y).float().mean().cpu().item()
        return loss, acc, logits
         
    def train(self, x_train, y_train, x_val, y_val):
        val_loss_old = torch.inf
        n_epochs_without_change = 0
        lrs_ = []
        iteration = 0
        
        while True:
            iteration += 1
            train_loss = self.train_step(x_train, y_train)
            val_loss, val_acc, _ = self.validate(x_val, y_val)
            self.scheduler.step(val_loss)
            n_epochs_without_change += int((val_loss_old - val_loss < self.tol) or (val_loss_old < val_loss)) 

            if (val_loss < val_loss_old):
                val_loss_old = val_loss
                n_epochs_without_change = 0
            if ((n_epochs_without_change == self.epochs_to_wait) and (iteration > self.ignore_first_n_steps)) or (iteration >= self.max_steps):
                break
            
        self.train_step(torch.cat([x_train, x_val], dim=0), torch.cat([y_train, y_val], dim=0))