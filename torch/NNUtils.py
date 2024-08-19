import torch
import torch.nn as nn
from torch.autograd import Variable


class Clipping(object):
    def __init__(self, cmin=0, cmax=1):
        self.cmin=cmin
        self.cmax=cmax

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(self.cmin,self.cmax)

class KernelMethod_SoftMax(nn.Module):
    def __init__(self, input_shape, centroids, widths, coeffs, resolution_const, resolution_scale, coeffs_clip,
                 train_centroids=False, train_widths=False, train_coeffs=True,
                 positive_coeffs=False,
                 name=None, **kwargs):
        super(KernelMethod_SoftMax, self).__init__()
        self.train_coeffs=train_coeffs
        self.train_centroids=train_centroids
        self.train_widths=train_widths
        self.centroids=centroids
        self.coeffs=coeffs
        self.widths=widths
        self.resolution_const=resolution_const
        self.resolution_scale=resolution_scale
        self.positive_coeffs=positive_coeffs
        if self.positive_coeffs:
            self.coeffs_clipping = Clipping(cmin=0, cmax=coeffs_clip)
        else:
            self.coeffs_clipping = Clipping(cmin=-1*coeffs_clip, cmax=coeffs_clip)
        self.coeffs = Variable(self.coeffs.reshape((-1, 1)).type(torch.double),
                               requires_grad=train_coeffs) # [M, 1]  
        self.kernel_layer = KernelLayer(input_shape=input_shape, centroids=self.centroids, widths=self.widths,
                                        train_centroids=self.train_centroids, train_widths=self.train_widths,
                                        resolution_const=self.resolution_const, resolution_scale=self.resolution_scale,
                                        name='kernel_layer')
        
    
    def call(self, x):
        K_x = self.kernel_layer.call(x) # [n, M]        
        W_x = self.coeffs  # [M, 1 ]
        Z = torch.sum(K_x, dim=1, keepdim=True) # [n, 1]
        out = torch.tensordot(K_x, W_x, dims=([1], [0])) # [n, 1]
        out = torch.divide(out, Z) # [n, 1]
        return out 

    def get_centroids_entropy(self):
        return self.kernel_layer.get_centroids_entropy()
    
    def get_coeffs(self):
        return self.coeffs

    def get_centroids(self):
        return self.kernel_layer.get_centroids()

    def get_widths(self):
        return self.kernel_layer.get_widths()

    def get_widths_tilde(self):
        return self.kernel_layer.get_widths_tilde()

    def clip_centroids(self):
        self.kernel_layer.clip_centroids()
        
    def clip_coeffs(self):
        self.coeffs_clipping(self.coeffs)

class KernelMethod(nn.Module):
    def __init__(self, input_shape, centroids, widths, coeffs, resolution_const, resolution_scale, coeffs_clip,
                 train_centroids=False, train_widths=False, train_coeffs=True,
                 positive_coeffs=False,
                 name=None, **kwargs):
        super(KernelMethod, self).__init__()
        self.train_coeffs=train_coeffs
        self.train_centroids=train_centroids
        self.train_widths=train_widths
        self.centroids=centroids
        self.coeffs=coeffs
        self.widths=widths
        self.resolution_const=resolution_const
        self.resolution_scale=resolution_scale
        self.positive_coeffs=positive_coeffs
        if self.positive_coeffs:
            self.coeffs_clipping = Clipping(cmin=0, cmax=coeffs_clip)
        else:
            self.coeffs_clipping = Clipping(cmin=-1*coeffs_clip, cmax=coeffs_clip)
        self.coeffs = Variable(self.coeffs.reshape((-1, 1)).type(torch.double),
                               requires_grad=train_coeffs) # [M, 1]  
        self.kernel_layer = KernelLayer(input_shape=input_shape, centroids=self.centroids, widths=self.widths,
                                        train_centroids=self.train_centroids, train_widths=self.train_widths,
                                        resolution_const=self.resolution_const, resolution_scale=self.resolution_scale,
                                        name='kernel_layer')
        
    
    def call(self, x):
        K_x = self.kernel_layer.call(x) # [n, M] 
        W_x = self.coeffs  # [M, 1 ]
        out = torch.tensordot(K_x, W_x, dims=([1], [0]))
        return out 

    def get_centroids_entropy(self):
        return self.kernel_layer.get_centroids_entropy()
    
    def get_coeffs(self):
        return self.coeffs

    def get_centroids(self):
        return self.kernel_layer.get_centroids()

    def get_widths(self):
        return self.kernel_layer.get_widths()

    def get_widths_tilde(self):
        return self.kernel_layer.get_widths_tilde()

    def clip_centroids(self):
        self.kernel_layer.clip_centroids()
        
    def clip_coeffs(self):
        self.coeffs_clipping(self.coeffs)
        
class KernelLayer(nn.Module):
    def __init__(self, input_shape, centroids, widths, resolution_const, resolution_scale,
                 cmin=0, cmax=1,train_centroids=False, train_widths=True,
                 name=None, **kwargs):
        super(KernelLayer, self).__init__()
        self.train_widths=train_widths
        self.train_centroids = train_centroids
        self.M = centroids.shape[0]
        self.resolution_const=Variable(resolution_const.reshape((-1,)).type(torch.double), requires_grad=False)
        self.resolution_scale=Variable(resolution_scale.reshape((-1,)).type(torch.double), requires_grad=False)
        self.clipping = Clipping(cmin=cmin, cmax=cmax)
        self.widths = Variable(widths.type(torch.double), requires_grad=train_widths)
        self.centroids = Variable(centroids.type(torch.double), requires_grad=train_centroids)
        
        
    def call(self, x):
        out = []
        widths = self.get_widths()#tf.math.add(tf.exp(self.widths), self.centroids*self.resolution_limit) # [M, d]  * relative resolution
        cov_diag = self.cov_diag(widths)
        consts = self.gauss_const(cov_diag) #[M, ]
        out = self.Kernel(consts, cov_diag, x)
        return out

    def get_widths(self): # transform width variable to account for resolution boudnaries (quadrature sum) 
        widths = torch.add(self.widths**2, self.resolution_const[None, :]**2) # [M, d]                                 
        widths+= torch.multiply(self.centroids, self.resolution_scale[None, :])**2 # [M, d]                         
        widths = torch.sqrt(widths) # [M, d]
        return widths
    
    def get_centroids(self):
        return self.centroids #[M, d]

    def clip_centroids(self):
        self.clipping(self.centroids)

    def get_centroids_entropy(self):
        """
        sum_j(sum_i(K_i(mu_j))*log(sum_i(K_i(mu_j))))
        return: scalar
        """
        K_mu = self.call(self.centroids) #[M, M]
        K_mu = torch.mean(K_mu, axis=1) # [M,]
        entropy = torch.sum(torch.multiply(K_mu, torch.log(K_mu)))
        return entropy
        
    def cov_diag(self, widths):
        return widths**2#torch.mul(widths, widths)#.double()
        
    def gauss_const(self, cov_diag):
        """                                                                                                       
        # widths.shape = [M, d]                                                                                   
        Returns the normalization constant for a gaussian                                                         
        # return.shape = [M,]                                                                                     
        """
        det_sigma_sq = torch.sum(cov_diag, axis=1)# [M,]                                                     
        return torch.sqrt(det_sigma_sq)/torch.pow(torch.sqrt(torch.tensor(2*torch.pi)), cov_diag.shape[1])

    def Kernel(self, consts, cov_diag, x):
        """                                                                                                             
        # x.shape = [N, d]                                                                                     
        # widths.shape = [M, d]                                                                                        
        # centroids.shape = [M, d]                                                                               
        Returns the gaussian function exponent term                                                           
        # return.shape = [N,M]                                                                                 
        """
        #cov_diag = torch.mul(widths, widths) #[M, d]                                                            
        dist_sq  = torch.subtract(x[:, None, :], self.centroids[None, :, :])**2 # [N, M, d]
        arg = torch.divide(dist_sq, cov_diag[None, :, :]) # [N, M, d]
        arg = -0.5*torch.sum(arg, axis=2) # [N, M]                                                            
        kernel = torch.multiply(consts[None, :], torch.exp(arg) ) #[N, M]                                
        return kernel # [N, M]

def NPLMLoss(true, pred):
    f   = pred[:, 0]
    y   = true[:, 0]
    w   = true[:, 1]
    return torch.sum((1-y)*w*(torch.exp(f)-1) - y*w*(f))

def L2Regularizer(pred):
    return torch.sum(torch.multiply(pred,pred))

def L1Regularizer(pred):
    return torch.sum(torch.abs(pred))

def CentroidsEntropyRegularizer(entropy):
    return entropy
