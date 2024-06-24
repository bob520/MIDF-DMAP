from torch import nn
import torch
import torch.nn.functional as F

class focal_loss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes=2, size_average='mean'):

        super(focal_loss,self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha,list):
            assert len(alpha)==num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha)

        self.gamma = gamma

        print('Focal Loss:')
        print('    Alpha = {}'.format(self.alpha))
        print('    Gamma = {}'.format(self.gamma))

    def forward(self,preds,labels):

        preds = preds.view(-1,preds.size(-1))
        alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        if self.size_average=='mean':
            loss = loss.mean()
        elif self.size_average=="BatchSize":
            loss = loss.squeeze()
        else:
            loss = loss.sum()
        return loss

if __name__ == '__main__':

    def test_focal_loss():
        num_classes = 2
        alpha = [0.25, 0.75]
        gamma = 2

        loss_fn = focal_loss(alpha=alpha, gamma=gamma, num_classes=num_classes,size_average="BatchSize")
        loss_fn2=torch.nn.CrossEntropyLoss(reduction='none')
        preds = torch.tensor([[0.2, 0.8], [0.6, 0.4]])
        print(preds.shape)
        labels = torch.tensor([1, 0])
        print(labels.shape)


        loss = loss_fn(preds, labels)

        print('Loss:', loss)
        print( loss.shape)

        loss2=loss_fn2(preds, labels)
        print(loss2)
        print(loss2.shape)


    test_focal_loss()
"""
不设置size_average="BatchSize"和reduction='none'时
focal Loss: tensor(0.0358)
CE loss：tensor(0.5178)

设置了时
focal Loss: tensor(0.0412，0.0303)
CE loss：tensor(0.4375，0.5981)
它们维度等于批次维度。各个数字的均值就是上面的数
"""