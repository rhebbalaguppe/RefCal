import torch
import torch.nn as nn
import logging
from libauc.losses import MultiLabelAUCMLoss

from .mmce import MMCE_weighted
from .flsd import FocalLossAdaptive
from .adafocal import AdaFocal

# from https://github.com/torrvision/focal_calibration/blob/main/Losses/focal_loss.py
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, **kwargs):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        logging.info("using gamma={}".format(gamma))

    def forward(self, input, target):

        target = target.view(-1,1)

        logpt = torch.nn.functional.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        
        return loss.mean()

class CrossEntropy(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(CrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input, target):
        return self.criterion(input, target)

class LogitMarginL1(nn.Module):
    """Add marginal penalty to logits:
        CE + alpha * max(0, max(l^n) - l^n - margin)

    Args:
        margin (float, optional): The margin value. Defaults to 10.
        alpha (float, optional): The balancing weight. Defaults to 0.1.
        ignore_index (int, optional):
            Specifies a target value that is ignored
            during training. Defaults to -100.

        The following args are related to balancing weight (alpha) scheduling.
        Note all the results presented in our paper are obtained without the scheduling strategy.
        So it's fine to ignore if you don't want to try it.

        schedule (str, optional):
            Different stragety to schedule the balancing weight alpha or not:
            "" | add | multiply | step. Defaults to "" (no scheduling).
            To activate schedule, you should call function
            `schedula_alpha` every epoch in your training code.
        mu (float, optional): scheduling weight. Defaults to 0.
        max_alpha (float, optional): Defaults to 100.0.
        step_size (int, optional): The step size for updating alpha. Defaults to 100.
    """
    def __init__(self,
                 margin: float = 10,
                 alpha: float = 0.1,
                 ignore_index: int = -100,
                 schedule: str = "",
                 mu: float = 0,
                 max_alpha: float = 100.0,
                 step_size: int = 100,**kwargs):
        super().__init__()
        assert schedule in ("", "add", "multiply", "step")
        self.margin = margin
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.mu = mu
        self.schedule = schedule
        self.max_alpha = max_alpha
        self.step_size = step_size

        self.cross_entropy = nn.CrossEntropyLoss()

    @property
    def names(self):
        return "loss", "loss_ce", "loss_margin_l1"

    def schedule_alpha(self, epoch):
        """Should be called in the training pipeline if you want to se schedule alpha
        """
        if self.schedule == "add":
            self.alpha = min(self.alpha + self.mu, self.max_alpha)
        elif self.schedule == "multiply":
            self.alpha = min(self.alpha * self.mu, self.max_alpha)
        elif self.schedule == "step":
            if (epoch + 1) % self.step_size == 0:
                self.alpha = min(self.alpha * self.mu, self.max_alpha)

    def get_diff(self, inputs):
        max_values = inputs.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        diff = max_values - inputs
        return diff

    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        loss_ce = self.cross_entropy(inputs, targets)
        # get logit distance
        diff = self.get_diff(inputs)
        # linear penalty where logit distances are larger than the margin
        loss_margin = nn.functional.relu(diff-self.margin).mean()
        loss = loss_ce + self.alpha * loss_margin

        return loss
        #return loss, loss_ce, loss_margin

class AUCML(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(AUCML, self).__init__()
        self.number_class = kwargs['args'].n_cls
        self.criterion = MultiLabelAUCMLoss(num_labels=kwargs['args'].n_cls)
        self.sm = nn.Softmax(dim=1)

    def forward(self, input, target):
        input = torch.sigmoid(input)
        target = nn.functional.one_hot(target,self.number_class)
        return self.criterion(input , target)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, alpha=0.0, dim=-1, **kwargs):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - alpha
        self.alpha = alpha
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        num_classes = pred.shape[self.dim]
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.alpha / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class MDCA(torch.nn.Module):
    def __init__(self):
        super(MDCA,self).__init__()

    def forward(self , output, target):
        output = torch.softmax(output, dim=1)
        # [batch, classes]
        loss = torch.tensor(0.0).cuda()
        batch, classes = output.shape
        for c in range(classes):
            avg_count = (target == c).float().mean()
            avg_conf = torch.mean(output[:,c])
            loss += torch.abs(avg_conf - avg_count)
        denom = classes
        loss /= denom
        return loss

class ClassficationAndMDCA(nn.Module):
    def __init__(self, loss="NLL+MDCA", alpha=0.1, beta=1.0, gamma=1.0, **kwargs):
        super(ClassficationAndMDCA, self).__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        if "NLL" in loss:
            self.classification_loss = nn.CrossEntropyLoss()
        elif "FL" in loss:
            self.classification_loss = FocalLoss(gamma=self.gamma)
        else:
            self.classification_loss = LabelSmoothingLoss(alpha=self.alpha) 
        self.MDCA = MDCA()

    def forward(self, logits, targets):
        loss_cls = self.classification_loss(logits, targets)
        loss_cal = self.MDCA(logits, targets)
        return loss_cls + self.beta * loss_cal

class BrierScore(nn.Module):
    def __init__(self, **kwargs):
        super(BrierScore, self).__init__()

    def forward(self, logits, target):
        
        target = target.view(-1,1)
        target_one_hot = torch.FloatTensor(logits.shape).to(target.get_device())
        target_one_hot.zero_()
        target_one_hot.scatter_(1, target, 1)

        pt = torch.softmax(logits, dim=1)
        squared_diff = (target_one_hot - pt) ** 2

        loss = torch.sum(squared_diff) / float(logits.shape[0])
        return loss

class DCA(nn.Module):
    def __init__(self, beta=1.0, **kwargs):
        super().__init__()
        self.beta = beta
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        output = torch.softmax(logits, dim=1)
        conf, pred_labels = torch.max(output, dim = 1)
        calib_loss = torch.abs(conf.mean() -  (pred_labels == targets).float().mean())
        return self.cls_loss(logits, targets) + self.beta * calib_loss

class MMCE(nn.Module):
    def __init__(self, beta=2.0, **kwargs):
        super().__init__()
        self.beta = beta
        self.mmce = MMCE_weighted()
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        cls = self.cls_loss(logits, targets)
        calib = self.mmce(logits, targets)
        return cls + self.beta * calib

class FLSD(nn.Module):
    def __init__(self, gamma=3.0, **kwargs):
        super().__init__()
        self.gamma = gamma
        self.criterion = FocalLossAdaptive(gamma=self.gamma)

    def forward(self, logits, targets):
        return self.criterion.forward(logits, targets)

class AdaptiveFocal(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.criterion = AdaFocal(args=self.args)

    def forward(self, logits, targets):
        return self.criterion.forward(logits, targets)

class  CRLLoss(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(CRLLoss, self).__init__()
        self.cls_criterion = nn.CrossEntropyLoss()
        self.ranking_criterion = nn.MarginRankingLoss(margin=0.0)
    
    def negative_entropy(data, normalize=False, max_value=None):
        softmax = nn.functional.softmax(data, dim=1)
        log_softmax = nn.functional.log_softmax(data, dim=1)
        entropy = softmax * log_softmax
        entropy = -1.0 * entropy.sum(dim=1)
        # normalize [0 ~ 1]
        if normalize:
            normalized_entropy = entropy / max_value
            return -normalized_entropy

        return -entropy

    def forward(self, input, target, args, idx,history):

        if args.rank_target == 'softmax':
            conf = nn.functional.softmax(input, dim=1)
            confidence, _ = conf.max(dim=1)
        # entropy
        elif args.rank_target == 'entropy':
            if args.dataset == 'cifar100':
                value_for_normalizing = 4.605170
            else:
                value_for_normalizing = 2.302585
            confidence = self.negative_entropy(input,normalize=True,max_value=value_for_normalizing)
        # margin
        elif args.rank_target == 'margin':
            conf, _ = torch.topk(nn.functional.softmax(input), 2, dim=1)
            conf[:,0] = conf[:,0] - conf[:,1]
            confidence = conf[:,0]
        
        # make input pair
        rank_input1 = confidence
        rank_input2 = torch.roll(confidence, -1)
        idx2 = torch.roll(idx, -1)

        # calc target, margin
        rank_target, rank_margin = history.get_target_margin(idx, idx2)
        rank_target_nonzero = rank_target.clone()
        rank_target_nonzero[rank_target_nonzero == 0] = 1
        rank_input2 = rank_input2 + rank_margin / rank_target_nonzero

        # ranking loss
        ranking_loss = self.ranking_criterion(rank_input1,
                                         rank_input2,
                                         rank_target)

        # total loss
        cls_loss = self.cls_criterion(input, target)
        ranking_loss = args.rank_weight * ranking_loss
        loss = cls_loss + ranking_loss
        
        return loss
    
class ACLSLoss(nn.Module):
    ## M is set to 6 for CIFAR-10 and 10 for other datasets. For CIFAR-100 as well, we are using 6.
    def __init__(self, lambda1=0.1, lambda2=0.01, M=6, dim=-1, **kwargs):
        super(ACLSLoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.M = M
        self.dim = dim

    def forward(self, logits, target, temp=1.0):
        pred = logits.log_softmax(dim=self.dim)
        #print(pred)
        num_classes = pred.shape[self.dim]

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            target_one_hot = nn.functional.one_hot(target, num_classes=num_classes)
            min_z_k = torch.min(logits, dim=self.dim, keepdim=True)[0]
            z_y = logits.gather(self.dim, target.unsqueeze(self.dim))  # Logit corresponding to the target class

            # Compute the ACLS loss for j = y^
            true_dist.scatter_(self.dim, target.unsqueeze(self.dim), (1 - self.lambda1 * nn.functional.relu(logits - min_z_k - self.M)))

            # Compute the ACLS loss for j ≠ y^
            true_dist_other = target_one_hot * self.lambda2 * nn.functional.relu(min_z_k - logits - self.M)
            true_dist += true_dist_other

        return temp * torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class LogitNormLoss(nn.Module):

    def __init__(self,temp,**kwargs):
        super(LogitNormLoss, self).__init__()
        self.device = "cuda"
        self.t = temp

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return nn.functional.cross_entropy(logit_norm, target)

loss_dict = {
    "focal_loss" : FocalLoss,
    "cross_entropy" : CrossEntropy,
    "LS" : LabelSmoothingLoss,
    "NLL+MDCA" : ClassficationAndMDCA,
    "LS+MDCA" : ClassficationAndMDCA,
    "FL+MDCA" : ClassficationAndMDCA,
    "brier_loss" : BrierScore,
    "NLL+DCA" : DCA,
    "MMCE" : MMCE,
    "FLSD" : FLSD,
    "AUCM_loss":AUCML,
    "MBLS":LogitMarginL1,
    "AdaFocal":AdaptiveFocal,
    "CRL":CRLLoss,
    "ACLS_loss":ACLSLoss,
    "logit_norm":LogitNormLoss
}