#-*- encoding: utf-8 -*-
import os, numpy
import re, codecs
from scipy import sparse

def fetchcorpuscontent(pathname,encoding=None):
    """fetch content of file in given pathname
    """
    corpus = []
    for filename in os.listdir(pathname):
        fullname = pathname+"/"+filename
        if os.path.isdir(fullname):
            corpus += fetchcorpuscontent(fullname)
        else:
            try:
                doc = codecs.open(fullname,mode="r",encoding=encoding).read()
                corpus.append(doc)
            except Exception as e:
                print e,filename
                continue
    return corpus
    
def listdir(pathname):
    """list filenames of given pathname
    """
    filelist = []
    if os.path.isdir(pathname):
        for filename in os.listdir(pathname):
            fullname = pathname+"/"+filename
            if os.path.isdir(fullname):
                filelist += listdir(fullname)
            else:
                filelist.append(fullname)
    return filelist

def fetchcorpus(pathname, encoding=None):
    """fetch file information in given pathname
    """
    corpus = dict()
    for filename in listdir(pathname):
        try:
            doc = codecs.open(filename,mode="r",encoding=encoding).read()
            corpus[filename] = doc
        except Exception as e:
            print e,filename
            continue
    return corpus
        
       
def loadcorpus(doclist, lowfreqthreshold=10,highfreqtopk=100, splitpattern=r"\s",wordpattern=r"\w.*\w"):
    """from raw doc list to matrix"""
    codebook = dict()
    indices, indptr = list(),[0,]
    for rawdoc in doclist:
        #string|->word
        for term in re.split(splitpattern,rawdoc):
            answer = re.match(wordpattern,term.lower())
            if answer is not None:
                indices.append(codebook.setdefault(answer.group(),len(codebook)))
                
        #data/indices/intptr for document
        indptr += [len(indices),]
    corpus = sparse.csr_matrix((numpy.ones_like(indices),indices,indptr),dtype="float32")
    #remove words while its frequence is too low/high
    freq = numpy.reshape(numpy.array(corpus.sum(0)),-1)
    valid = freq > lowfreqthreshold
    if highfreqtopk > 0:
        for idx in freq.argsort()[-highfreqtopk:]:
            valid[idx] = False
    #corpus
    corpus = corpus[:,valid]
    #codebook(word\->code) to vocab(code\->word)
    code2word = dict(zip(codebook.values(),codebook.keys()))
    vocab = [code2word[k] for k, isvalid in enumerate(valid) if isvalid]
    return corpus, vocab
    
def loadcorpus2(doclist, lowfreqthreshold=10, highfreqtopk=100, splitpattern=r"\s",wordpattern=r"\w.*\w"):
    """from raw doc list to matrix"""    
    codebook = dict(); pattern = '\w.*\w'
    data, indices, indptr = list(), list(),[0,]
    for rawdoc in doclist:
        #string|->word
        observation = []
        for term in re.split(splitpattern,rawdoc):
            answer = re.match(pattern, term.lower())
            if answer is not None:
                observation.append(codebook.setdefault(answer.group(),len(codebook)))
        
        freq = sparse.csr_matrix((\
            numpy.ones_like(observation),\
            (numpy.zeros_like(observation),observation)\
            ),dtype="float32")
        #data/indices/intptr for document
        data += freq.data.tolist()
        indices += freq.nonzero()[1].tolist()
        indptr += [len(data),]
    corpus = sparse.csr_matrix((data,indices,indptr),dtype="float32")
    #remove words while its frequence is too low/high
    freq = numpy.reshape(numpy.array(corpus.sum(0)),-1)
    valid = freq > lowfreqthreshold
    if highfreqtopk > 0:
        for idx in freq.argsort()[-highfreqtopk:]:
            valid[idx] = False
    #corpus
    corpus = corpus[:,valid]
    #codebook(word\->code) to vocab(code\->word)
    code2word = dict(zip(codebook.values(),codebook.keys()))
    vocab = [code2word[k] for k, isvalid in enumerate(valid) if isvalid]
    return corpus, vocab
    
def loadvocab(filepath):
    """indexing machinery from integer to word label"""
    fp = file(filepath)
    vocab = numpy.array(\
        filter(\
            None,\
            [record.split('\t')[0] for record in fp.read().split("\r\n")]\
        )\
    )
    return vocab
    

#####################################################
"""By experiment, we know that it takes 3 mins for pdf file with 8 pages downloading,
   1.5mins for pdf file parsing, 10 secs for word encoding and bowing.
"""   
#import re, os
#from vb.fileop import load_file, save_file
#from nltk import word_tokenize
#
#class encoded_texts(object):
#    def __init__(self, texts=None):
#        if not texts:
#            self.vocab = []
#            self.texts = []
#            return
#            
#        #1.tokens of input texts
#        word_of_text = [word_tokenize(text.lower()) for text in texts]
#        
#        #2.vocabulary
#        words = []
#        for text in word_of_text:
#            words.extend(text)
#        self.vocab = list(set(words))
#        
#        #3.replace words of text with vocabulary index
#        self.texts = []
#        for text  in word_of_text:
#            self.texts.append([self.vocab.index(word) for word in text])
#     
#     
#    def load(self, corpus_path =None, text_seperator=r"\n", word_seperator=r" "):
#        #1.change environment
#        if corpus_path:
#            current_dir = os.getcwd()
#            os.chdir(corpus_path) 
#            
#        #2.load file
#        vocab_file_name = "vocab.txt"    
#        for file_name in os.listdir(os.getcwd()):
#            if file_name == vocab_file_name:
#                #2.1 vocabulary
#               self.vocab = load_file(file_name, None, word_seperator)
#            else:
#                #2.2 text
#                self.texts = load_file(file_name, text_seperator, word_seperator, None)
#        
#        #3.restore environment
#        if corpus_path:
#            os.chdir(current_dir)
#        
#     
#     
#    def save(self, corpus_path=None, text_seperator="\n", word_seperator=r" "):
#        #1.change environment
#        if corpus_path:
#            current_dir = os.getcwd()
#            os.chdir(corpus_path)
#            
#        #2.save vocabulary
#        vocab_file_name = "vocab.txt"
#        save_file(self.vocab, vocab_file_name, word_seperator)
#
#        #3.save text
#        text_list = [
#            word_seperator.join("%s"%word for word in text)
#                for text in self.texts
#                ]
#        file_name = "corpus.txt"
#        save_file(text_list, file_name, text_seperator) 
#        
#        #4.restore environment
#        if corpus_path:
#            os.chdir(current_dir)
#
#
#class bow(object):
#    def __init__(self, texts=None):
#        if not texts:
#            self.vocab = []
#            self.texts = []
#            return        
#
#        #1.tokens of input texts
#        word_of_text = [word_tokenize(text.lower()) for text in texts]
#        #2.vocabulary
#        words = []
#        for text in word_of_text:
#            words.extend(text)
#        self.vocab = list(set(words))
#        #3.replace words of text with vocabulary index
#        #3.1 initialize texts as list
#        self.texts = []
#        for words in word_of_text:
#            #3.2 for every text, compute frequence of tokens
#            #Note: we can use list.count to complete the task like this
#            #for word in self.vocab: text[word] = tokens.count(word)
#            #but the method is inefficient
#            text = {}
#            for word in words:
#                keyword = self.vocab.index(word)
#                text[keyword] = text.get(keyword,  0) +1
#            #3.3 collection of text frequency
#            self.texts.append(text)            
#        
#        
#    def load(self, corpus_path=None, vocab_seperator=r" ", text_seperator="\n", word_seperator=r"[ :]"):
#        #1.change environment
#        if corpus_path:
#            current_dir = os.getcwd()
#            os.chdir(corpus_path) 
#            
#        #2.load file
#        vocab_file_name = "vocab.txt"    
#        for file_name in os.listdir(os.getcwd()):
#            if file_name == vocab_file_name:
#                #2.1 vocabulary
#               self.vocab = load_file(file_name, None,vocab_seperator)
#            else:
#                #2.2 text
#                self.texts = [dict(zip(raw_text[1::2], raw_text[2::2]))
#                    for raw_text in load_file(file_name, text_seperator, word_seperator, int)]
#        
#        #3.restore environment
#        if corpus_path:
#            os.chdir(current_dir)
#        
#        
#    def save(self, corpus_path=None, text_seperator="\n", word_seperator=r" ", key_value_seperator=r":"):
#        #1.change environment
#        if corpus_path:
#            current_dir = os.getcwd()
#            os.chdir(corpus_path)
#            
#        #2.save vocabulary
#        vocab_file_name = "vocab.txt"
#        save_file(self.vocab, vocab_file_name, word_seperator)
#
#        #3.save text
#        text_list = [
#            word_seperator.join(["%d"%len(text)] + ["%s%s%d"%(key, key_value_seperator, value) for key, value in text.items()])
#                for text in self.texts
#                ]
#        file_name = "corpus.txt"
#        save_file(text_list, file_name, text_seperator) 
#        
#        #4.restore environment
#        if corpus_path:
#            os.chdir(current_dir)
            
if __name__ == '__main__':
    #path->list
    pathname = os.getcwd() +"/../../data/corpus/"
    datasetname = "sogouC(reduced)"
    #corpus = fetchcorpuscontent(pathname+datasetname, encoding="GBK")
    corpus = fetchcorpus(pathname+datasetname,encoding="GBK")
    print len(corpus), corpus.values()[0]    
    corpus, vocab = loadcorpus(corpus.values(), splitpattern=u"[^\0x80-\0xff]")
#    from scipy import io
#    io.savemat(pathname+corpusname,{"corpus":corpus,"vocab":vocab})
    
    #list->mat
#    from sklearn import datasets
#    dataset = datasets.fetch_20newsgroups(subset='all',shuffle=False)
#    gnd = dataset.target.reshape(-1,1)
#    corpus, vocab = loadcorpus(dataset.data, highfreqtopk=0)
#    corpusname = "20newsgroups"
#    io.savemat(pathname+corpusname,{"corpus":corpus, "vocab":vocab, "gnd":gnd})
#    
#    topk = 100; corpus, vocab = loadcorpus(dataset.data, highfreqtopk=topk)
#    corpusname = "20newsgroups(freq_top%d_removed)"%(topk)
#    io.savemat(pathname+corpusname,{"corpus":corpus,"vocab":vocab, "gnd":gnd})