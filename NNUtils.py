import tensorflow as tf
from tensorflow import Variable, Module
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.models import Model#, VariableModel
from tensorflow.keras.constraints import Constraint

import numpy as np

class Clipping(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range                                                                                                       
    '''
    def __init__(self, cmin=0, cmax=1):
        self.cmin = cmin
        self.cmax = cmax
        
    def __call__(self, p):
        return tf.clip_by_value(p, clip_value_min=self.cmin, clip_value_max=self.cmax)
    

def NPLMLoss(true, pred):
    f   = pred[:, 0]
    y   = true[:, 0]
    w   = true[:, 1]
    return tf.reduce_sum((1-y)*w*(tf.exp(f)-1) - y*w*(f))

def WeightedCrossEntropyLoss(true, pred):
    f   = pred[:, 0]
    y   = true[:, 0]
    w   = true[:, 1]
    return tf.reduce_sum((1-y)*w*tf.math.log(1+tf.exp(f)) + y*w*tf.math.log(1+tf.exp(-1*f)))

def L2Regularizer(pred):
    return tf.reduce_sum(tf.multiply(pred,pred))


def GaussianRegularizer(pred):
    return tf.reduce_sum(tf.exp(-1*tf.multiply(pred,pred)))

def CentroidsAttractiveRegularizer(target, data, centroids, widths):
    y = target[:, 0:1] # [N, 1]
    w = target[:, 1:2] # [N, 1]
    widths_sq = tf.multiply(widths, widths) # [M, d]                                                              
    widths_sq = tf.reduce_sum(widths_sq, axis=1) # [M,]
    distance_sq = tf.math.subtract(tf.expand_dims(data,axis=1),tf.expand_dims(centroids,axis=0))**2 # [N, M, d]
    distance_sq = tf.reduce_sum(distance_sq, axis=2) # [N, M]
    distance_sq_y = tf.multiply(distance_sq, (2*y-1)) # [N, M]
    distance_sq_wy= tf.multiply(distance_sq_y, w) # [N, M]  
    ratio = tf.divide(distance_sq_wy, tf.expand_dims(widths_sq,axis=0)) # [N, M]                                     
    ratio_data_sum= tf.abs(tf.reduce_sum(y*ratio, axis=0)/tf.reduce_sum(y)) #[M,]                                         
    return tf.reduce_mean(ratio_data_sum) # sum over centroids                                                    

def CentroidsRepulsiveRegularizer(centroids, widths):
    distance_sq_components = tf.math.subtract(tf.expand_dims(centroids,axis=0),tf.expand_dims(centroids,axis=1))**2 # [ N, M, d]      
    distance_sq_sum = tf.reduce_sum(distance_sq_components, axis=2) # [M, M]                                      
    widths_sum_components = tf.math.add(tf.expand_dims(widths, axis=0), tf.expand_dims(widths, axis=1)) # [M, M, d]    
    widths_sq_components = tf.multiply(widths_sum_components, widths_sum_components) # [M, M, d]                  
    widths_sq_sum = tf.reduce_sum(widths_sq_components, axis=2) # [M,M]                                           
    ratio = tf.divide(distance_sq_sum, widths_sq_sum) # [M, M]                                                    
    return -1*tf.reduce_mean(ratio)*0.5

def CentroidsEntropyRegularizer(entropy):
    return entropy

class KernelMethod(Module):
    def __init__(self, input_shape, centroids, widths, coeffs, resolution_const, resolution_scale, coeffs_clip,
                 train_centroids=False, train_widths=False, train_coeffs=True,
                 positive_coeffs=False,
                 name=None, **kwargs):
        super().__init__(name=name, **kwargs)
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

    #def __build__(self, input_shape):
        self.coeffs = Variable(initial_value=self.coeffs.reshape((-1, 1)), dtype="float64",
                               trainable=self.train_coeffs, name='coeffs') # [M, 1]                                                                        
        self.kernel_layer = KernelLayer(input_shape=input_shape, centroids=self.centroids, widths=self.widths,
                                        train_centroids=self.train_centroids, train_widths=self.train_widths,
                                        resolution_const=self.resolution_const, resolution_scale=self.resolution_scale,
                                        name='kernel_layer')
        
    
    def call(self, x):
        K_x = self.kernel_layer.call(x) # [n, M]                                                                       
        W_x = self.coeffs  # [M, 1 ]
        #if self.positive_coeffs: W_x = tf.exp(W_x)
        out = tf.tensordot(K_x, W_x, axes=[1, 0])
        return out 

    def get_centroids_entropy(self):
        return tf.cast(self.kernel_layer.get_centroids_entropy(), dtype=tf.float64)
    
    def get_coeffs(self):
        return tf.cast(self.coeffs, dtype=tf.float64)

    def get_centroids(self):
        return tf.cast(self.kernel_layer.get_centroids(), dtype=tf.float64)

    def get_widths(self):
        return tf.cast(self.kernel_layer.get_widths(), dtype=tf.float64)

    def get_widths_tilde(self):
        return tf.cast(self.kernel_layer.get_widths_tilde(), dtype=tf.float64)

    def clip_centroids(self):
        self.kernel_layer.clip_centroids()
        
    def clip_coeffs(self):
        self.coeffs.assign(self.coeffs_clipping(self.coeffs))
        
class KernelLayer(Module):
    def __init__(self, input_shape, centroids, widths, resolution_const, resolution_scale,
                 cmin=0, cmax=1,train_centroids=False, train_widths=True,
                 name=None, **kwargs):
        self.widths = widths
        self.centroids = centroids
        self.train_widths=train_widths
        self.train_centroids = train_centroids
        self.M = centroids.shape[0]
        self.resolution_const=tf.constant(resolution_const.reshape((-1,)), dtype="float64")
        self.resolution_scale=tf.constant(resolution_scale.reshape((-1,)), dtype="float64")
        self.clipping = Clipping(cmin=cmin, cmax=cmax)
        #self.positive = tf.keras.layers.ReLU()
        super().__init__(name=name, **kwargs)
        
    #def __build__(self, input_shape):
        self.widths = Variable(initial_value=self.widths, dtype="float64", trainable=self.train_widths, name='widths')
        self.centroids = Variable(initial_value=self.centroids, dtype="float64", trainable=self.train_centroids, name='centroids')
        
    def call(self, x):
        out = []
        widths = self.get_widths()#tf.math.add(tf.exp(self.widths), self.centroids*self.resolution_limit) # [M, d]  * relative resolution
        consts = self.gauss_const(widths) #[M, ]                                                                 
        out = self.Kernel(consts, widths, x)
        return out

    def get_widths(self): # transform width variable to account for resolution boudnaries (quadrature sum)                         
        widths = tf.math.add(self.widths**2, tf.expand_dims(self.resolution_const, axis=0)**2) # [M, d]                                 
        widths+= tf.math.multiply(self.centroids, tf.expand_dims(self.resolution_scale, axis=0))**2 # [M, d]                         
        widths = tf.sqrt(widths) # [M, d]                                                                                 
        return widths
    
    def get_centroids(self):
        return self.centroids #[M, d]

    def clip_centroids(self):
        self.centroids.assign(self.clipping(self.centroids))

    def get_centroids_entropy(self):
        """
        sum_j(sum_i(K_i(mu_j))*log(sum_i(K_i(mu_j))))
        return: scalar
        """
        K_mu = self.call(self.centroids) #[M, M]
#        log_K_mu = tf.math.log(K_mu) #[M, M]
        K_mu = tf.reduce_mean(K_mu, axis=1) # [M,]
#        log_K_mu = tf.reduce_sum(log_K_mu, axis=1) # [M,]
        entropy = tf.reduce_sum(tf.multiply(K_mu, tf.math.log(K_mu)))
        return entropy
    
    def gauss_const(self, widths):
        """                                                                                                       
        # widths.shape = [M, d]                                                                                   
        Returns the normalization constant for a gaussian                                                         
        # return.shape = [M,]                                                                                     
        """
        cov_diag = tf.cast(tf.multiply( widths, widths), dtype=tf.float64) #[M, d]                                                           
        det_sigma_sq = tf.cast(tf.reduce_sum(cov_diag, axis=1), dtype=tf.float64) # [M,]                                                     
        return tf.sqrt(det_sigma_sq)/tf.math.pow(tf.cast(tf.sqrt(2*np.pi), dtype=tf.float64), cov_diag.shape[1])

    def Kernel(self, consts, widths, x):
        """                                                                                                             
        # x.shape = [N, d]                                                                                     
        # widths.shape = [M, d]                                                                                        
        # centroids.shape = [M, d]                                                                               
        Returns the gaussian function exponent term                                                           
        # return.shape = [N,M]                                                                                 
        """
        cov_diag = tf.multiply(widths, widths) #[M, d]                                                            
        dist_sq  = tf.math.subtract(tf.cast(tf.expand_dims(x,axis=1), dtype=tf.float64),
                                    tf.cast(tf.expand_dims(self.centroids,axis=0), dtype=tf.float64))**2 # [N, M, d]                                                        
        arg = tf.divide(dist_sq, tf.expand_dims(cov_diag, axis=0)) # [N, M, d]                                    
        arg = -0.5*tf.reduce_sum(arg, axis=2) # [N, M]                                                            
        kernel = tf.multiply(tf.expand_dims(consts, axis=0), tf.exp(arg) ) #[N, M]                                
        return kernel # [N, M]  

class BumpHuntModel(Module):
    def __init__(self, input_shape, centroids_x, widths_x, coeffs_x,
                 centroids_m, widths_m, coeffs_m, norm_m, resolution_limit,
                 train_centroids_x=False, train_widths_x=False, train_coeffs_x=True,
                 train_centroids_m=False, train_widths_m=False, train_coeffs_m=True,
                 train_norm=False, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.d = len(resolution_limit)
        self.x_discriminator = KernelMethod((None, input_shape[1]-1),
                                            centroids_x, widths_x, coeffs_x,
                                            resolution_limit[:self.d-1],
                                            train_centroids=train_centroids_x,
                                            train_widths=train_widths_x,
                                            train_coeffs=train_coeffs_x,
                                            name='x_discriminator')
        
        self.bumphunt_finder = KernelMethod((None, 1),
                                            centroids_m, widths_m, coeffs_m,
                                            resolution_limit[self.d-1:],
                                            train_centroids=train_centroids_m,
                                            train_widths=train_widths_m,
                                            train_coeffs=train_coeffs_m,
                                            positive_coeffs=True,
                                            name='bumphunt_finder')
        
        self.norm_finder = Variable(initial_value=norm_m.reshape((-1, 1)), dtype="float64", trainable=train_norm, name='norm') #[1,1]
        self.norm_finder_solo = Variable(initial_value=norm_m.reshape((-1, 1)), dtype="float64", trainable=train_norm, name='norm_solo') #[1,1] 

    def call(self, x):
        m = x[:, -1:]
        X = x[:, :-1]
        out_discr = self.x_discriminator(X)
        out_bumph = tf.math.add(self.bumphunt_finder(m), self.norm_finder)
        out_flath = self.norm_finder_solo*tf.ones_like(out_bumph)
        return tf.keras.layers.Concatenate(axis=1)([out_bumph, out_discr, out_flath])
    
    def get_centroids_x(self):
        return tf.cast(self.x_discriminator.get_centroids(), dtype=tf.float64)
    
    def get_widths_x(self):
        return tf.cast(self.x_discriminator.get_widths(), dtype=tf.float64)
    
    def get_coeffs_x(self):
        return tf.cast(self.x_discriminator.get_coeffs(), dtype=tf.float64)
    
    def get_centroids_m(self):
        return tf.cast(self.bumphunt_finder.get_centroids(), dtype=tf.float64)

    def get_widths_m(self):
        return tf.cast(self.bumphunt_finder.get_widths(), dtype=tf.float64)
    
    def get_coeffs_m(self):
        return tf.cast(self.bumphunt_finder.get_coeffs(), dtype=tf.float64)

    def get_norm_m(self):
        return tf.cast(self.norm_finder, dtype=tf.float64)

    def get_norm_solo(self):
        return tf.cast(self.norm_finder_solo, dtype=tf.float64)
    
    def clip_centroids_m(self):
        self.bumphunt_finder.clip_centroids()
        
    def clip_centroids_x(self):
        self.x_discriminator.clip_centroids()

class Adam(tf.Module):

    def __init__(self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, ep=1e-7):
      # Initialize the Adam parameters
      self.beta_1 = tf.cast(beta_1, dtype=tf.float64)
      self.beta_2 = tf.cast(beta_2, dtype=tf.float64)
      self.learning_rate = tf.cast(learning_rate, dtype=tf.float64)
      self.ep = tf.cast(ep, dtype=tf.float64)
      self.t = tf.cast(1., dtype=tf.float64)
      self.v_dvar, self.s_dvar = [], []
      self.title = f"Adam: learning rate={self.learning_rate}"
      self.built = False

    def apply_gradients(self, grads, vars):
      # Set up moment and RMSprop slots for each variable on the first call
      if not self.built:
        for var in vars:
          v = tf.Variable(tf.zeros(shape=var.shape, dtype=tf.float64))
          s = tf.Variable(tf.zeros(shape=var.shape, dtype=tf.float64))
          self.v_dvar.append(v)
          self.s_dvar.append(s)
        self.built = True
      # Perform Adam updates
      for i, (d_var, var) in enumerate(zip(grads, vars)):
        # Moment calculation
        self.v_dvar[i] = self.beta_1*self.v_dvar[i] + (1-self.beta_1)*d_var
        # RMSprop calculation
        self.s_dvar[i] = self.beta_2*self.s_dvar[i] + (1-self.beta_2)*tf.square(d_var)
        # Bias correction
        v_dvar_bc = self.v_dvar[i]/(1-(self.beta_1**self.t))
        s_dvar_bc = self.s_dvar[i]/(1-(self.beta_2**self.t))
        # Update model variables
        var.assign_sub(self.learning_rate*(v_dvar_bc/(tf.sqrt(s_dvar_bc) + self.ep)))
      # Increment the iteration counter
      self.t += 1.

class GradientDescent(tf.Module):

  def __init__(self, learning_rate=1e-3):
    # Initialize parameters
    self.learning_rate = learning_rate
    self.title = f"Gradient descent optimizer: learning rate={self.learning_rate}"

  def apply_gradients(self, grads, vars):
    # Update variables
    for grad, var in zip(grads, vars):
      var.assign_sub(self.learning_rate*grad)
        