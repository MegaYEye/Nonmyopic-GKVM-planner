import numpy as np
import torch
import gpytorch
import time

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, config):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())
        self.covar_module.base_kernel.lengthscale=config["kernel_scale"]
        self.likelihood.noise = config["kernel_noise"]
        # print("lengthscale:", self.covar_module.base_kernel.lengthscale.item()) # do not know how to specify the lengthscale
        # print("noise:",self.likelihood.noise.item())
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPR(object):
    def __init__(self, init_X, init_y, config):
        self.X = torch.FloatTensor(init_X).cuda()
        self.y = torch.FloatTensor(init_y).cuda()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()#.double()
        self.model = ExactGPModel(self.X, self.y, self.likelihood, config).cuda()#.double()
        self.model.train()
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        self.optimizer =  torch.optim.Adam([{'params': self.model.parameters()}], lr=0.01)
        
    def cpu(self):
        self.model = self.model.cpu()
        self.X = self.X.cpu()
        self.y = self.y.cpu()
        
    def cuda(self):
        self.model = self.model.cuda()
        self.X = self.X.cuda()
        self.y = self.y.cuda()
        
    def add_new_observation(self, new_observe_X, new_observe_y):
        """
            Add new observation (incremental). It is said to be faster than set_train_data 
        """
        X = torch.FloatTensor(new_observe_X).cuda()
        y = torch.FloatTensor(new_observe_y).cuda()

        
        self.X = torch.cat([self.X,X], dim=0)
        self.y = torch.cat([self.y,y], dim=0)
        self.model.set_train_data(self.X, self.y, strict=False)  
 
        
    def set_observation(self, X, y):
        """
            Change the training data. 
        """
        self.X = torch.FloatTensor(X).cuda()
        self.y = torch.FloatTensor(y).cuda() 
        with torch.no_grad():
            self.model.set_train_data(self.X, self.y, strict=False)   
            
    def downsample(self,amp=2):
        X = self.X.cpu().numpy()
        X = X * amp
        X = np.round(X)
        X, index = np.unique(X, axis=0,return_index=True)
        index = torch.LongTensor(index)
        # print(index)
        X = X.astype(np.float32)/amp
        y = self.y.cpu().numpy()[index]
        self.set_observation(X,y)
        
        
    
    def train(self, training_iter=1, verbose=True):
        self.model.train()
        self.likelihood.train()
        for i in range(training_iter):
            self.optimizer.zero_grad()
            output = self.model(self.X)
            loss = -self.mll(output, self.y)
            loss.backward()
            if verbose:
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, training_iter, loss.item(),
                    self.model.covar_module.base_kernel.lengthscale.item(),
                    self.model.likelihood.noise.item()
                ))
            self.optimizer.step()
        return loss.detach().cpu().item()
    
    def test(self, X):
        self.model.eval()
        self.likelihood.eval()
        prediction_buf = []
        uncertainty_buf = []
        isplit=0
        step_length = 5000
        with torch.no_grad(),\
            gpytorch.settings.fast_pred_var(),\
            gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False), \
            gpytorch.settings.max_cg_iterations(3000), \
            gpytorch.settings.eval_cg_tolerance(1e-1),\
            gpytorch.settings.memory_efficient(True),\
            gpytorch.settings.max_preconditioner_size(80):

            while isplit*step_length<X.shape[0]:
                isplit += 1
                split_min = step_length * (isplit-1)
                split_max = np.minimum(step_length * isplit, X.shape[0])
                
                xstar_tensor = torch.FloatTensor(X[split_min:split_max,:]).cuda()
                p = self.model(xstar_tensor) # most of the time, here
                observed_pred = self.likelihood(p) # first test takes longer time.
                
                prediction = observed_pred.mean.cpu().numpy()
                prediction_buf.append(prediction)
                confidence_lower, confidence_upper = observed_pred.confidence_region()
                confidence_lower = confidence_lower.cpu().numpy()
                confidence_upper = confidence_upper.cpu().numpy()
                uncertainty_buf.append(confidence_upper - confidence_lower)

        prediction = np.hstack(prediction_buf)
        uncertainty = np.hstack(uncertainty_buf)
        uncertainty = np.sqrt(uncertainty) 
        return prediction, uncertainty
    

        
if __name__ == '__main__':
    X = np.random.randn(10000,3)
    y = np.random.randn(10000)
    gp = GPR(X,y)
    gp.train(10)
