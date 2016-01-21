# -*- encoding: utf-8 -*-
###########################################################
##         construct texts by extracting information from assigned url address
##Note:
##1.By experiment, we found the most time consume step is download, which is 6 min
##   in case kunzhou.net, while parser take 1.5 min for a pdf file. 
##2.In cs.nyu.edu/roweis data, the pdf file organized with page header, tailer, note, image,figure,
##   mathematics expression,and reference, which is also what i did.
###########################################################
import os, sys
from datetime import datetime
from bs4 import BeautifulSoup
import urllib2
from StringIO import StringIO
from fileop import pdf_file, save_file
import re
from nltk import *

filetype = r".pdf"
archor_tag = r"a"
href_attr = r"href"
abs_url_pattern = r"http://"
url_seperator = r"/"

def texts_from_url(root_addr, filter =lambda filename: filename.endswith(filetype), parser=pdf_file, save_dir=None):
    """extract file address from the assigned root_addr"""
    #1.initialization the archor container
    #1.1 connect to the assigned root_addr
    if not root_addr.endswith(url_seperator):
       root_addr.append(url_seperator)
    html = urllib2.urlopen(root_addr).read()
    #1.2 archor container
    archors = BeautifulSoup(html).find_all(archor_tag)
    
    #2.read file
    #2.1 get url addr of file.
    urls = set(archor[href_attr].startswith(url_seperator) and 
                   root_addr + archor[href_attr][len(url_seperator):] or root_addr + archor[href_attr]
                    for archor in archors
                        if archor.has_key(href_attr) and filter(archor[href_attr]))     
    #2.2 set current work directory    
    if save_dir:      
        current_dir = os.getcwd()
        os.chdir(save_dir)
    #2.3 save the raw file and extract text 
    texts = []
    for url_addr in urls:
        ###########For debug##############
        start_time = datetime.now()
        ################################
        
        #2.3.1 download the file and connect with stream        
        instream = StringIO()
        instream.write(urllib2.urlopen(url_addr).read()) 
        print("\ttime of download:%s"%(datetime.now()-start_time))
        #2.3.2 extract text using parser
        texts.append(parser(instream).read()) 
        print("\ttime of parse:%s"%(datetime.now()-start_time))
        #2.3.3 save the download file
        if save_dir:
            fp = file(url_addr.split(url_seperator)[-1], "wb")
            fp.write(instream)
            fp.close()
            
        ###########For debug##############
        print("collapsed time for %s processing is %s"%(url_addr, str(datetime.now()-start_time)))
        ################################
    #2.4 restore work directory
    if save_dir:
        os.chdir(current_dir)
        
    return texts
        

#####################################################
##          construct texts by extracting information from local file system
#####################################################
#texts_from_local_fs(dir_name,lambda filename:filename,lambda fp:fp)
def texts_from_local_fs(dir_name, filter = lambda filename: filename.endswith(filetype), parser=pdf_file):
    #1.recursive exitance
    if not dir_name:
        return []
        
    #2. text information collection from current_dir 
    #2.0 change directory
    current_dir = os.getcwd()
    os.chdir(dir_name)
    #2.1 get filenames of dir_name
    filenames = (filename for filename in os.listdir(os.getcwd()) if os.path.isfile(filename) and filter(filename))
    #2.2 initialize using the content of assigned files
    ##by experiment, we know it takes 1 mininus for extracting text from pdf with 8 pages. 
    last_time = datetime.now()
    texts = []
    for filename in filenames:
        texts.append(parser(file(filename, "rb")).read())
        current_time = datetime.now()
        print("it takes %s time for extracting %s"%(current_time - last_time, filename))
        last_time = current_time
    #texts = [parser(file(filename, "rb")).read() for filename in filenames]
    
    #3. recursive collection of texts in child_dir
    for childdir in os.listdir(os.getcwd()):
        if os.path.isdir(childdir):
            texts.append(texts_from_fs(childdir))
    
    #4.restore the directory and return value
    os.chdir(current_dir)
    return texts

#####################################################
##          save the texts information as .txt files
#####################################################
def save_as_text(texts, save_dir=None):        
    for id, text in enumerate(texts):
        file_name = "%.4d.txt"%(id) 
        save_file(text, file_name, "")
    
    
if __name__ == "__main__":
    #url_texts = texts_from_url(r"http://www.kunzhou.net/")
    local_texts = texts_from_local_fs(r"C:\Users\hutch\Desktop\beal03.pdf")