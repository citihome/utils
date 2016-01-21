# -*- coding: utf-8 -*-
"""
Deep Neural Networks
"""

import theano, numpy
from theano import tensor, sparse


def euclid2ProbAsterik(L2_input, threshold):
    L2_input = tensor.clip(L2_input-L2_input.max(),  -threshold, 0)
    return tensor.sum(tensor.exp(L2_input))

def euclid2LogicAsterik(L2_input, threshold):
    L2_input = tensor.clip(L2_input, -threshold, threshold)
    return tensor.log(1+tensor.exp(L2_input))
    
def euclid2SignAsterik(L2_input, threshold):
    L2_input = tensor.clip(L2_input, -threshold, threshold)
    return 2*tensor.log(1+tensor.exp(L2_input)) - L2_input
    
    
class ConvolutionalPooledLayer(object):
    def __init__(self, rng, P_input, L2_input, kernelshape, downscale):
        #parameter definition/initialization
        #1.parameter of Layer
        self.L2_w = theano.shared(\
            rng.uniform(low=-1,high=1,size=kernelshape).astype(theano.config.floatX)\
        )
        self.L2_b = theano.shared(numpy.zeros(kernelshape[0], dtype=theano.config.floatX))
        self.params = [self.L2_w, self.L2_b]
        #2.output definition
        #convolution operation
        downsample_input = tensor.nnet.conv2d(\
            self.P_input,\
            self.L2_w,\
            border_mode='half', \
            filter_flip=True\
        )
        if L2_input is not None:
            downsample_input += tensor.nnet.conv2d(\
                self.L2_input,\
                tensor.tanh(self.L2_w),\
                border_mode='half',\
                filter_flip=True\
            )
        #downsample+bias operation
        self.L2_output = tensor.signal.downsample.max_pool_2d(\
            downsample_input, downscale, ignore_border=True\
        )+self.L2_b.dimshuffle('x',0,'x','x')
        self.P_output = tensor.tanh(self.L2_output)
        
        #layer_training and predict model definition
        #L2_output->Pasterik_output
        threshold = tensor.scalar("threshold")
        Pasterik_output = 2*tensor.log(\
            1+tensor.exp(tensor.clip(self.L2_output, -threshold, threshold))\
        )-self.L2_output 
        #difference between output and operator
        d_xw_w2 = tensor.mean(Pasterik_output) - \
            0.5*(tensor.sum(tensor.mean(self.L2_w**2, 0))+tensor.mean(self.L2_b**2))
        #gradient/updates
        eta = tensor.scalar("eta")#variable representing learning rate
        grads = theano.grad(d_xw_w2, self.params)
        updates = [(param, param+eta*grad) for param, grad in zip(self.params, grads)]
        #Layer training model definition
        self._train = theano.function(\
            inputs=[P_input, eta, threshold],outputs=d_xw_w2,updates=updates\
        )
        self._predict = theano.function(inputs=[P_input],outputs=self.P_output)
        
    
    
class CNN(object):
    """ Convolutional Neural Network. Also remember the elements are connected with sum operation
    """
    def __init__(self, rng, P_input, L2_input=None, **kwargs):
        """CNN based connectism.symbol definition
        """        
        ## symbol definition and parameter initialization
        self.P_input = tensor.tensor4('x') if P_input is None else P_input
        #2.parameter of networks
        shape = kwargs.get("shape") or [(4,16,16,2,2),(4,4,4,2,2)] 
        self.connectism = list(); self.params = list()
        stack_size = 1
        for k, (kernel_num, kernelshape0, kernelshape1, downscale0, downscale1) in enumerate(shape):
            Lk = ConvolutionalPooledLayer(\
                rng,\
                P_input=P_input,\
                L2_input=L2_input,\
                kernelshape=(kernel_num, stack_size, kernelshape0, kernelshape1),\
                downscale=(downscale0, downscale1)
            )
            self.connectism.append(Lk)
            self.params += Lk.params
            #parameter for next layer
            P_input, L2_input = Lk.P_output, Lk.L2_output#take output of current layer as input of next layer
            stack_size = kernel_num#also kernel num of underlied layer works as stack size of next layer 
        #3.output of network: flatten on the output of the last layer.
        self.P_output = Lk.P_output.flatten(2)
        self.L2_output = Lk.L2_output.flatten(2)
        
    def build(self, logsumexp=euclid2ProbAsterik):
        """building training/prediction models
        """
        #training and predict model definition
        #L2_output->Pasterik_output
        threshold = tensor.scalar("threshold")
        Pasterik_output = logsumexp(self.L2_output, threshold)
        eta = tensor.scalar("eta", dtype=theano.config.floatX)#variable representing learning rate
        
        #1.objective function: take guidInfo as proposition-like measure
        y = tensor.vector("y",dtype=theano.config.floatX)
        Likelihood = tensor.sum(\
            tensor.mean(y.dimshuffle(0,'x')*Pasterik_output, axis=0)\
        )
        #gradient/updates
        grads = theano.grad(Likelihood, self.params)
        updates = [(param, param+eta*grad) for param, grad in zip(self.params, grads)]
        #supervised learning model
        self._supervise = theano.function(\
            inputs=[self.P_input, y, eta, threshold],\
            outputs=Likelihood,\
            updates=updates\
        )
        #difference between output and operator
        d_xw_w2 = tensor.mean(Pasterik_output)
        for param in self.params:
            d_xw_w2 -= 0.5*tensor.sum(tensor.mean(param**2, 0))
        #gradient/updates
        grads = theano.grad(d_xw_w2, self.params)
        updates = [(param, param+eta*grad) for param, grad in zip(self.params, grads)]
        #unsupervised learning model
        self._unsupervise = theano.function(\
            inputs=[self.P_input, eta, threshold],\
            outputs=d_xw_w2,\
            updates=updates\
        )        
        #3.predict model
        self._predict = theano.function([self.P_input], (self.L2_output>=0))
        
    def pretrain(self, x, eta=0.01, n_iter=100, batch_size=100, verbose=True, **kwargs):
        """hierarchical pretrain
        """
        for k, Lk in enumerate(self.connectism):
            for t in xrange(n_iter):
                L = 0
                for s in xrange(0,x.shape[0]-batch_size, batch_size):
                    L += Lk._train(x[s:s+batch_size], eta)
                L += Lk._train(x[s:],eta)
                if verbose:
                    print "Layer %d"%(k),\
                          "d(wx, w2)_(iter=%d)=%.4f)"%(t, L/numpy.ceil(x.shape[0]/batch_size))
            ##
            x = Lk._predict(x)#output of Lk=input of next layer
     
    def fit(self, x, y=None, eta=0.01, threshold=10, n_iter=100, batch_size=100, verbose=True, **kwargs):
        """hierarchical pretrain
        """
        for t in xrange(n_iter):
            L = 0
            for s in xrange(0,x.shape[0]-batch_size, batch_size):
                L += self._unsupervise(x[s:s+batch_size], eta, threshold) if y is None else\
                    self._supervise(x[s:s+batch_size], y[s:s+batch_size], eta, threshold)
            L += self._unsupervise(x[s:], eta, threshold) if y is None else\
                self._supervise(x[s:], y[s:], eta, threshold)
            if verbose:
                print "likelihood(iter=%d)=%.4f)"%(t, L/numpy.ceil(x.shape[0]/batch_size))
    
    def predict(self, x, **kwargs):
        """prediction for batch data"""
        y = self._predict(x)
        return y
        
        
class SparseCNN(CNN):
    """attach sparse node in the fgraph of CNN
    """
    def __init__(self, rng, P_input, L2_input=None, **kwargs):
        #1.symbol declaration, initialization and definition
        I = sparse.csr_matrix("I") if P_input is None else P_input
        shape = kwargs.get("shape") or [(16,1,32,32), (4,16,16,2,2), (4,4,4,2,2)]
        dict_size, kwargs["shape"] = shape[0], shape[1:]
        D = theano.shared(\
            rng.uniform(low=-1,high=1,size=dict_size).astype(theano.config.floatX)\
        )
        DI = sparse.dot(I, D)#array access=dot operation
        
        #2.attaches I and D into the fgraph
        super(SparseCNN, self).__init__(rng=rng, P_input=DI, **kwargs)
        self.params += [D,]
        self.P_input = I#take I as input for the sparseCNN
        
class SequenceCNN(CNN):
    """CNN for sequence data/conditional random field processing.
    """
    def __init__(self, rng, P_input, L2_input=None, **kwargs):
        #symbol declaration, initialization and definition
        x_1_tm1, x_t = (\
                tensor.tensor4("x_1_tm1",dtype=theano.config.floatX),\
                tensor.tensor4("x_t",dtype=theano.config.floatX)\
            )\
            if P_input is None else P_input[:2]
        
        #elements of history
        super(SequenceCNN, self).__init__(rng=rng, P_input=x_1_tm1, L2_input=L2_input, **kwargs)#attaches new elements into the fgraph
        self.L2_output_1_tm1 = self.L2_output
        
        #elements of current time
        self.L2_output_t = theano.clone(self.L2_output_1_tm1, replace={x_1_tm1:x_t})
        
        #element prepartion for model building
        self.P_input = (x_1_tm1,x_t)
        self.L2_output = self.L2_output_1_tm1*self.L2_output_t
        
    def build(self):
        """building training/prediction models
        """
        super(SequenceCNN, self).build()        
        self.codebook_1_tm1 = theano.function([self.P_input[0]], (self.L2_output_1_tm1>=0))
        self.codebook_t = theano.function([self.P_input[1]], (self.L2_output_t>=0))

class SparseSequenceCNN(SparseCNN):
    #SequenceCNN(SparseCNN):#
    """CNN for sequence data/conditional random field processing.
    """
    def __init__(self, rng, P_input, L2_input, **kwargs):
        #symbol declaration, initialization and definition
        x_1_tm1, x_t = (\
                sparse.csr_matrix("x_1_tm1", dtype=theano.config.floatX),\
                sparse.csr_matrix("x_t",dtype=theano.config.floatX)\
            )\
            if P_input is None else P_input[:2]
        
        #elements of history
        shape = kwargs.get("shape")
        if shape is not None:
            dict_size = shape[0]
            if len(shape) <= 1:
                del shape["shape"]
            else:
                shape["shape"] = shape["shape"][1:]
        else:
            dict_size = (16,1,32,32)
        D_1_tm1 = theano.shared(rng.normal(size=dict_size).astype(theano.config.floatX))        
        Dx_1_tm1 = sparse.dot(x_1_tm1, D_1_tm1)#array access=dot operation      
        super(SequenceCNN, self).__init__(rng=rng, inputsymbol=Dx_1_tm1, **kwargs)#attaches new elements into the fgraph
        self.L2_output_1_tm1 = self.L2_output
        
        #elements of current time
        D_t = theano.shared(rng.normal(size=dict_size).astype(theano.config.floatX))        
        Dx_t = sparse.dot(x_t, D_t)#array access=dot operation
        self.L2_output_t = theano.clone(self.L2_output_1_tm1, replace={Dx_1_tm1:Dx_t})
        
        #element prepartion for model building
        self.P_input = (x_1_tm1,x_t)
        self.params += [D_1_tm1, D_t]
        self.L2_output = self.L2_output_1_tm1*self.L2_output_t
        
    def build(self):
        """building training/prediction models
        """
        super(SparseSequenceCNN, self).build()        
        self.codebook_1_tm1 = theano.function([self.P_input[0]], (self.L2_output_1_tm1>=0))
        self.codebook_t = theano.function([self.P_input[1]], (self.L2_output_t>=0))
        
class RegressionCNN(CNN):
      #RegressionCNN(SparseCNN):#
      #RegressionCNN(SequenceCNN):#
    """CNN for regression problem"""
    def __init__(self, rng, P_input, L2_input=None, **kwargs):
        if P_input is None:
            y = tensor.vector("y", dtype=theano.config.floatX)
        else:
            y = P_input[-1]
            P_input = None if len(P_input) <= 1 else P_input[:-1]        
        
        super(RegressionCNN, self).__init__(rng, P_input, **kwargs)#fgraph construction        
        self.L2_output *= y.dimshuffle(0,'x')#for model building
        
    def build(self):
        """redefine predict function"""
        super(RegressionCNN, self).build()
        self._predict = theano.function([self.P_input], tensor.sum(self.P_output*self.L2_output,1))