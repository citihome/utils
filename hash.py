#-*- encoding: utf-8 -*-

import numpy
from itertools import izip

class LSH(object):
    """Locality sensitive hashing for euclidean distance
    """
    def __init__(self, rng=None, **kwargs):
        """define key2value function
        """
        if rng is None:
            rng = numpy.random.RandomState(numpy.random.randint(2**32-1))
        self.rng = rng
        
    def value2key(self, x, **kwargs):
        """define value2key function
        """
        #1.make sure x is organized as instance x value dimensional
        if x.ndim == 1:
            x = x.reshape(x.size, -1)
        elif x.ndim > 2:
            x = x.reshape(x.shape[0],-1)
        
        #hash function initialization
        #in the first place, we need to transform x to N(0,1) by (x-mu)/sigma
        if not hasattr(self, "hash_s") or x.shape[1] != self.hash_s.shape[0]:
            delta, epsilon = kwargs.get("delta") or 1e-2, kwargs.get("epsilon") or 0.01
            #the equation for scale, hashnum, delta and epsilon:
            #phi(scale*hashnum*delta)=epsilon=>scale*hashnum*delta = log(1/epsilon-1)
            scale_hashnum = (numpy.log2(1./epsilon-1)-1)/delta
            hashnum = numpy.ceil(numpy.sqrt(scale_hashnum))
            scale = scale_hashnum/hashnum
                
            mu,sigma = x.mean(), numpy.sqrt(x.var())
            distance = kwargs.get("distance") or "l2"
            if distance == "l1":
                #uniform distribution
                self.hash_scale = self.rng.uniform(high=scale,size=(x.shape[1],hashnum))
                
            elif distance == "l2":
                #guassian distribution
                self.hash_scale = self.rng.normal(scale=1.47/(delta*sigma),size=(x.shape[1],hashnum))
            else:
                return None
            ##
            self.hash_bias = self.hash_scale.sum(0)*mu
        
            #memory system
            self.ID2key = dict()#ID to its location
            self.key2value = dict()#key2value:storage
            self.stamp = 0
            
        
        #2. from value to key
        keys = numpy.ceil(numpy.dot(x,self.hash_scale)-self.hash_bias)
        return keys
        
    def buildHash(self, values, data=None, IDs=None, **kwargs):
        """build hash for given values.
        """
        
        keys = self.value2key(values, **kwargs)#val2key
        if data is not None:
            values = data
        
        #1.id allocation
        if IDs is None or IDs.size != values.shape[0]:
            needIDAlloc = True
            IDs = numpy.empty(values.shape[0])
        else:
            needIDAlloc = False
        for t, (key, value) in enumerate(izip(keys,values)):
            if needIDAlloc:
                IDs[t] = self.stamp
            self.stamp += 1
            #2.array2str
            key = key.tostring()
            self.ID2key[IDs[t]] = key
            #3.key2val
            if self.key2value.get(key) is None:
                self.key2value[key] = {IDs[t]:value}
            else:
                self.key2value[key][IDs[t]] = value
        return IDs
    
    
    def remove(self, IDs):
        """remove given ids in its database
        """
        for ID in IDs:
            key = self.ID2key(ID)
            if self.key2val[key].get(ID) is not None:
                del self.key2val[key][ID]
        
     
    def reply(self, query,**kwargs):
        """reply all values w.r.t the query 
        """
        if query.shape[0] != 1:
            query = numpy.expand_dims(query, 0)
            
        key = self.value2key(query).tostring()#value2key
        #value is attached with ID in memory system
        bucket = self.key2value.get(key)
        return [] if bucket is None else bucket.values()


class ArgmaxCosine(LSH):
    """argmax_{data in D} theta(query, data)
    """
    def value2key(self, x, **kwargs):
        
        #1.make sure x is organized as instance x value dimensional
        if x.ndim == 1:
            x = x.reshape(x.size, -1)
        elif x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        
        if not hasattr(self, "hash") or x.shape[1] != self.hash.shape[0]:
            #P(hash value equals|theta, hashnum)=(1-theta/pi)**hashnum
            #int_P(hash value equals|theta<delta, hashnum)
            #using Laplace's asymptotic integral method, the probability function
            #is approximated by (1-delta/pi)^{hashnum+1}
            #log2(1-delta/pi)*(hashnum+1)<=log2(eps)
            #hashnum=log2(eps)/log2(1-delta/pi)-1
            #it shows that 
            # delta, eps |-> hashnum
            #(pi/180,5e-5)|-> 826
            #(pi/180,1e-5)|-> 537
            #(pi/36, 1e-2)|-> 163
            #(pi/36, 5e-2)|-> 106
            #(pi/18, 1e-2)|->  80
            #(pi/18, 5e-2)|->  52
            mu, sigma = x.mean(), numpy.sqrt(x.var())
            delta, epsilon = kwargs.get("delta") or numpy.pi/18, abs(kwargs.get("epsilon") or 5e-2)
            hashnum = numpy.ceil(numpy.log2(epsilon)/numpy.log2(1-delta/numpy.pi)-1)
            self.hash_scale = self.rng.normal(scale=sigma,size=(x.shape[1],hashnum))
            self.hash_bias = self.hash_scale.sum(0)*mu
            
            #memory system
            self.ID2key = dict()#ID to its location
            self.key2value = dict()#key2value:storage
            self.stamp = 0
        
        #2. from value to key
        keys = ((numpy.dot(x,self.hash_scale)-self.hash_bias)>=0).astype("int8")
        return keys


class MIP(LSH):
    """asymetric LSH based MIP, where we transform the MIP problem into LSH problem
    """        
    def buildHash(self, x, data=None, IDs=None, **kwargs):
        
        if data is None:
            data = x
            
        #1.make sure x is organized as instance x value dimensional
        if x.ndim == 1:
            x = x.reshape(x.size, -1)
        elif x.ndim > 2:
            x = x.reshape(x.shape[0],-1)
            
        #scaling data with nrm2 is less than 1.
        nrm2 = (x*x).sum(-1)
        if not hasattr(self, "frac_1_normalizer"):
            self.scale = 0.99
            self.frac_1_normalizer = self.scale/(nrm2.max())
        x *= self.frac_1_normalizer
        nrm2 *= self.frac_1_normalizer
        
        if not hasattr(self, "hash_order"):
            #{nrm2.max()=scale}**{2**ext_size} <= delta
            #ext_size=log2(log_{scale}(delta))=log2(log2(delta)/log2(scale))
            delta = kwargs.get("delta") or 1e-2
            ext_size = numpy.ceil(numpy.log2(numpy.log2(delta)/numpy.log2(self.scale)))
            #kwargs["delta"] = delta + 0.25*ext_size
            self.nrm2_order = 0.5*numpy.cumprod(2*numpy.ones((1,ext_size)))#for hashing
            self.query_ext = 0.5*numpy.ones((1,ext_size+1))#for query
            self.nrm2_offset = 0.5*ext_size
        #nrm2**(2**k) k=1,\cdots, ext_size
        nrm2_ext = nrm2[:,numpy.newaxis]**self.nrm2_order
        xprime = numpy.hstack([x, nrm2_ext])
        
        #by the underlied buildHash
        return super(MIP,self).buildHash(xprime, data, IDs, **kwargs)
    
    def reply(self, query, **kwargs):
        query = numpy.hstack([self.frac_1_normalizer*query.reshape(1,-1), self.query_ext])
        return super(MIP, self).reply(query, **kwargs)


class ArgminHammingDistance(LSH):
    """ Locality Sensitive Hasing for Hamming Distance, 
        we introduce probability P[not equals to x] to provide event happends with 1/2,
        thus the codes is discriminative.
    """
    def value2key(self, x, **kwargs):
        assert x.dtype == "int32"
        
        #1. In the first place, we setup random basis of linear space
        if not hasattr(self,"hashIdx"):
            delta, epsilon = kwargs.get("delta") or 1e-2, kwargs.get("epsilon") or 1e-2
            #P(hash value equals|d, hashnum) = (1-d)**hashnum
            #P(hash value equals|d>=delta, hashnum)=(1-delta)**(hashnum+1)/(hashnum+1)<eps
            #hashnum >= log_{1-delta}(eps)
            hashnum = numpy.ceil(numpy.log2(epsilon)/numpy.log2(delta))
            
            x_samplesize, x_dimlen = x.shape
            self.place = self.rng.rand_int(high=x_dimlen, size=hashnum)
            self.pattern = numpy.empty(hashnum,dtype="int32")
            self.defaultvalue = self.rng.rand_int(high=1, size=hashnum)
            for k, dim, defaultvalue in enumerate(izip(self.place,self.defaultvalue)):
                sample0 = self.rng.rand_int(x_samplesize)
                if defaultvalue == 0:
                    #with probability P[value of dim==x[sample0, dim]]
                    self.pattern[k] = x[sample0, dim]
                    
                else:
                    #with probability P[value of dim!=x[sample0, dim]]
                    while True:
                        sample1 = self.rng.rand_int(x_samplesize)
                        if sample1 != sample0:
                            break
                        else:
                            sample0 = sample1
                    self.pattern[k] = x[sample1, dim]
                    
        #2. then we put x into linear space
        values = numpy.int((x[:,self.place]==self.pattern)==self.defaultvalue)
        return values
        
    
class ArgminEditDistance(LSH):
    """Locality sensitive Hashing for Editing Distance, there operate is support,
       a) match
       b) insert
       c) delete
    """
    def value2key(self, x, **kwargs):
        assert x.dtype == "int32"
        
        #1. In the first place, we setup random basis of linear space
        if not hasattr(self,"hashIdx"):
            delta, epsilon = kwargs.get("delta") or 1e-2, kwargs.get("epsilon") or 1e-2
            #P(hash value equals|d, hashnum) = (1-d)**hashnum
            #P(hash value equals|d>=delta, hashnum)=(1-delta)**(hashnum+1)/(hashnum+1)<eps
            #hashnum >= log_{1-delta}(eps)
            hashnum = numpy.ceil(numpy.log2(epsilon)/numpy.log2(1-delta))
            
            x_samplesize, x_dimlen = x.shape
            self.place = self.rng.rand_int(high=x_dimlen, size=hashnum)
            self.pattern = numpy.empty(hashnum, dtype="int32")
            self.defaultvalue = numpy.empty(hashnum, dtype="bool")
            for k, dim in enumerate(self.place):
                sample0, sample1 = self.rng.rand_int(high=x_samplesize,size=2)
                if sample0 == sample1:
                    #with probability P[value of dim==x[sample0, dim]]
                    self.pattern[k], self.defaultvalue[k] = x[sample0, dim],False
                    
                else:
                    #with probability P[value of dim==x[sample0, dim]] 
                    #and P[value of dim !=x[sample1,dim]] seperately
                    self.pattern[k], self.defaultvalue[k] =\
                    (sample0, False)\
                    if self.rng.randint(high=2) == 0 else\
                    (sample1, True)
                    
        #2. then we put x into linear space: 
        #if matches the pattern, the weight of basis<-default value
        values = numpy.int((x[:,self.place]==self.pattern)==self.defaultvalue)
        return values
        

from scipy import sparse
class MinHash(ArgminHammingDistance):
    """argmin of Jaccard distance, used for dimension reduction: 
       large scale sparse matrix->low compact dimension.
       we can also take any LSH tech as the underlied module for search.
    """
    def value2key(self, x, **kwargs):        
        """the key point is that the hash function is shared by all sets, and 
           probability of element be selected is monopolied by its owner's 
           configuration. permutation of the value dimension+index of first 
           nonzero element is the one.
        """        
        #1.in the first place, make sure x is organized as instance x value dimension
        if x.ndim == 1:
            x = x.reshape(x.size, -1)
        elif x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
            
        #2.we then gives sparse x/enumerate elements compact/uniform code
        assert sparse.issparse(x)
        if not hasattr(self, "codebook") or x.shape[1] != self.hash.shape[0]:
            #P[hash code equals|Jaccard distance d]=(1-d)**codebooknum
            #P[hash code equals|d>delta]=delta**(codebooknum+1)/(codebooknum+1)<eps
            #hashnum=log_{delta}(eps)
            delta, epsilon = kwargs.get("delta") or 0.05, kwargs.get("epsilon") or 0.01
            codebook_num = numpy.ceil(numpy.log2(epsilon)/numpy.log2(1-delta))
            #random permutation
            self.codebook = numpy.vstack([numpy.random.permutation(x.shape[1]) for cnt in xrange(codebook_num)])
        #the overload is very heavy
        x1 = numpy.vstack([\
                     numpy.array([\
                         xt.nonzero()[1][randp[xt.nonzero()[1]].argmin()]\
                         for randp in self.codebook\
                     ])\
                     for xt in x\
                 ])
        #3.although we gives compact code for input, but the framework for search has
        #not setup. so in the last place, we prepare anything for search. 
        values = super(MinHash, self).value2key(x1, **kwargs)        
        return values
                 
    
    def buildHash(self, x, data=None, **kwargs):
        """we assume the x is sparse.csr_matrix
        """
        #very long sparse vector to an compact vector/high level value2key 
        values = self.codec(x)
        #from value to key: first index of nonzero elements.
        if data is None:
            data = x
        return super(MinHash, self).build_hash(values, data, **kwargs)
        
    def reply(self, query,**kwargs):
        return super(MinHash, self).reply(self.codec(query))
        
        
class GLSH(object):
    """ Generalized LSH
        we approximate given distance function with the first-order taylor expansion
        Fhat(query) = F(data_0) + <data, dF(data_0, query)>
                    = <[F(data_0); data], [1, dF(data_0)(query)]>
        which corresponds to MiniMax inner product on two things
        min_data max_data0 <F(data_0)+data, 1+dF(data_0)(query)>
        **I wander if we can solve minimax problem by MIP?
        
        If we use the second-order taylor expansion
        Fhat(query)=F0+<data,dF0(query)>+data*data':d2F0(query)
        while there is multiply operator for dF0, d2F0, the computational overhead
        is too heavy.
        **I wander if there is techniques to replace the operation data*data':d2F0)
    """
    def __init__(self, rng, F, dF, **kwargs):
        kwargs["rng"] = rng
        self.kwargs=kwargs
        self.F, self.dF = F, dF            
        
    def buildHash(self, values, data=None, **kwargs):
        """build hash for given values.
        """
        if not hasattr(self,"factory"):
            sample_num = kwargs.get("sample_num") or 50
            self.factory = [(MIP(self.rng, self.kwargs), self.F(data0), self.dF0(data0))\
                for data0 in self.rng.shuffle(values)[:sample_num]]
        
        IDs = None
        for Hash, F0, dF0 in self.factory:
            IDs = Hash.buildHash(numpy.vstack([F0, values]), IDs=IDs, **kwargs)
        
    def remove(self, IDs):
        """remove given ids in its database
        """
        for Hash,F0,dF0 in self.factory:
            Hash.remove(IDs)
                    
    def reply(self, query,**kwargs):
        #MIP based Minimax problem solution
        return [Hash.reply(numpy.vstack([1,dF0(query)])) for Hash, F0, dF0 in self.factory]