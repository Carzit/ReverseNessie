from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions.gumbel import Gumbel
from torch.distributions.log_normal import LogNormal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.negative_binomial import NegativeBinomial
from torch.distributions.normal import Normal
from torch.distributions.poisson import Poisson
from torch.distributions.weibull import Weibull

# It can be called Trajectory


'''
Q_predict obtains the weights and corresponding probability distribution parameters of the neural network model output 
to construct a estimate probability distribution .

model_out --[weights and distribution paras]--> Q_predict
Q_predict(copy_number) = probility
'''

class DistributionMixture(nn.Module):
    '''
    估计边缘分布的基类

    初始化时传入相应概率分布参数
    由神经网络预测的k个分布组合而成

    被调用时传入相应x值返回对应
    '''
    def __init__(self, weights:torch.Tensor) -> None:# model_out shape:[batch_size, components_size, distribution_params_size]
        super(DistributionMixture, self).__init__()
        self.batch_size: int = weights.shape[0]
        self.component_size: int = weights.shape[1]
        self.w: torch.Tensor = weights
        self.copy_size: int
        self.distribution: torch.distributions.distribution.Distribution
    
    def forward(self, X:torch.Tensor):#X shape=[batch_size, copy_size]
        self.copy_size = X.shape[1]
        self.distribution = self.distribution.expand([self.copy_size, self.batch_size, self.component_size])#[copy_size,batch_size, component_size]
        self.probs = torch.exp(self.distribution.log_prob(X.transpose(0,1).unsqueeze(-1))).permute(1,2,0)#[batch_size, component_size, copy_size]
        self.w = self.w.unsqueeze(-1).expand(self.batch_size, self.component_size, self.copy_size)
        return (self.w * self.probs).sum(1) #->[batch_size, copy_size]
    
class MultivariateDistributionMixture(nn.Module):
    '''
    估计联合分布的基类

    初始化时传入相应概率分布参数
    由神经网络预测的k个分布组合而成

    被调用时传入相应(x,y,...)值返回对应
    '''
    def __init__(self, model_out:torch.Tensor) -> None:# model_out shape:[batch_size, components_size, distribution_params_size]
        super(MultivariateDistributionMixture, self).__init__()
        self.batch_size: int = model_out.shape[0]
        self.component_size: int = model_out.shape[1]
        self.variate_size: int
        self.copy_size: int
        
        self.w: torch.Tensor
        self.distribution: torch.distributions.distribution.Distribution
    
    def forward(self, X:torch.Tensor):#X shape=[batch_size, copy_size, variate_size]， where copy_size=copy_size_per_variate^variate_size
        self.copy_size = X.shape[1]
        self.variate_size = X.shape[2]
        self.distribution = self.distribution.expand([self.copy_size, self.batch_size, self.component_size])#[copy_size, batch_size, component_size]
        self.probs = torch.exp(self.distribution.log_prob(X.transpose(0,1).unsqueeze(2))).permute(1,2,0)#[batch_size, component_size, copy_size]
        self.w = self.w.unsqueeze(-1).expand(self.batch_size, self.component_size, self.copy_size)
        return (self.w * self.probs).sum(1) #->[batch_size, copy_size]


class DistributionMixture_NegativeBinomial(DistributionMixture):
    '''
    估计分布
    由神经网络预测的k个负二项分布组合而成
    __init__输入形状:[batch_size, components_size, 3]
    __call__输入形状:[batch_size, copy_size]
    '''
    def __init__(self, weights:torch.Tensor, r:torch.Tensor, p:torch.Tensor) -> None:
        super().__init__(weights)
        self.distribution = NegativeBinomial(r, p)


class DistributionMixture_Poisson(DistributionMixture):
    '''
    估计分布
    由神经网络预测的components_size个泊松分布组合而成
    __init__输入形状:[batch_size, components_size, 2]
    __call__输入形状:[batch_size, copy_size]
    '''
    def __init__(self, weights:torch.Tensor, lamda:torch.Tensor) -> None:
        super().__init__(weights)
        self.distribution = Poisson(lamda)

class DistributionMixture_Normal(DistributionMixture):
    '''
    估计分布
    由神经网络预测的components_size个正态分布组合而成
    __init__输入形状:[batch_size, components_size, 2]
    __call__输入形状:[batch_size, copy_size]
    '''
    def __init__(self, weights:torch.Tensor, mean:torch.Tensor, sd:torch.Tensor) -> None:
        super(DistributionMixture, self).__init__(weights)
        self.distribution = Normal(loc=mean, scale=sd)

class DistributionMixture_LogNormal(DistributionMixture):
    '''
    估计分布
    由神经网络预测的components_size个对数正态分布组合而成
    __init__输入形状:[batch_size, components_size, 2]
    __call__输入形状:[batch_size, copy_size]
    '''
    def __init__(self, weights:torch.Tensor, mean:torch.Tensor, sd:torch.Tensor) -> None:
        super(DistributionMixture_LogNormal, self).__init__(weights)
        self.distribution = LogNormal(loc=mean, scale=sd)

    def forward(self, X:torch.Tensor):#X shape=[batch_size, copy_size]
        X = nn.Softplus()(X) # 对数正态分布X=0时不存在对应概率，softplus处理使为正
        return super(DistributionMixture_LogNormal, self).forward(X)

class DistributionMixture_Weibull(DistributionMixture):
    '''
    估计分布
    由神经网络预测的components_size个Weibull分布组合而成
    __init__输入形状:[batch_size, components_size, 2]
    __call__输入形状:[batch_size, copy_size]
    '''
    def __init__(self, weights:torch.Tensor, scale:torch.Tensor, concentration:torch.Tensor) -> None:
        super(DistributionMixture_Weibull, self).__init__(weights)
        self.distribution = Weibull(scale=scale, concentration=concentration)

    def forward(self, X:torch.Tensor):#X shape=[batch_size, copy_size]
        X = nn.Softplus()(X) # Weibull分布X=0时不存在对应概率，softplus处理使为正
        return super(DistributionMixture_Weibull, self).forward(X)

class DistributionMixture_Gumbel(DistributionMixture):
    '''
    估计分布
    由神经网络预测的components_size个Gumbel分布组合而成
    __init__输入形状:[batch_size, components_size, 2]
    __call__输入形状:[batch_size, copy_size]
    '''
    def __init__(self, weights:torch.Tensor, loc:torch.Tensor, scale:torch.Tensor) -> None:
        super(DistributionMixture_Gumbel, self).__init__(weights)
        self.distribution = Gumbel(loc=loc, scale=scale)

class DistributionMixture_MultivariateNormal2D(MultivariateDistributionMixture):
    '''
    估计分布
    由神经网络预测的components_size个二元正态分布组合而成
    __init__输入形状:[batch_size, components_size, 6], 最后1维上的顺序为权重w, x1均值, x2均值, x1方差, x2方差, x1和x2的协方差
    __call__输入形状:[batch_size, copy_size], 其中 copy_size=copy_size_per_variate^variate_size
    '''

    def __init__(self, model_out:torch.Tensor) -> None:
        super(MultivariateDistributionMixture, self).__init__(model_out)
        self.w, mean1, mean2, sd1, sd2, corr = model_out.permute((2,0,1))
        mean_matrix = torch.stack([mean1, mean2], dim=2) #[batch_size, component_size, 2]
        cov_matrix = torch.stack([torch.stack([sd1*sd1, corr*sd1*sd2]), torch.stack([corr*sd1*sd2, sd2*sd2])]).permute(2,3,1,0)#[batch_size, component_size, 2, 2]

        self.distribution = MultivariateNormal(loc=mean_matrix, covariance_matrix=cov_matrix)