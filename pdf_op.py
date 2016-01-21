# -*- encoding: utf-8 -*-
#####################################################
##                   txt file format read and write
#####################################################
import re
import sys
def load_file(file_name, text_seperator, word_seperator, operator=None):
    try:
        #1. ->text list
        fp = file(file_name, "rb")
        if text_seperator:
            text_spliter = re.compile(text_seperator)
            raw_texts = text_spliter.split(fp.read())
        else:
            raw_texts = [fp.read()]
        fp.close()
        
        #2. ->word list
        word_spliter = re.compile(word_seperator)
        if operator:
            texts = [[operator(word) for word in word_spliter.split(text)]
                        for text in raw_texts]
        else:
            texts = [word_spliter.split(text) for text in raw_texts]
            
        #3.If there is only one element, reduce the dimension
        if len(texts) == 1:
            return texts[0]
        return texts
    except :
        print("failed in %s call", sys._getframe().f_code.co_name)


def save_file(word_list, file_name, word_seperator):
    if word_list:
        fp = file(file_name, "wb")
        fp.write(word_seperator.join(word_list))
        fp.close()
    
    
#####################################################
##                    construct text from assigned pdf stream
#####################################################
from pdfminer.pdfparser import PDFDocument,PDFParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter, process_pdf
from pdfminer.converter import TextConverter, PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTFigure, LTImage

class pdf_file(object):
    """text factory from pdf format file"""
    def __init__(self, fp, obj_seperator="\n", page_seperator="\n"):
        self.fp = fp
        self.obj_seperator = obj_seperator
        self.page_seperator = page_seperator
    
    def read(self,fp=None):
        """extract text from assigned pdf file"""
        def parse_lt_objs(objs):
            """parse literal object: objects|->str"""
            def update_hash_table(table, obj, pct=0.2):
                """insert element into hash table"""
                #1.get keyword for hash
                x0 = obj.bbox[0]
                x1 = obj.bbox[2]
                #2.search location in (hash)table
                key_found = False
                for key, value in table.items():
                    if ((x0 >= (1-pct)*key[0] and x0 <= (1+pct)*key[0])
                        and (x1 >= (1-pct)*key[1] and x1 <= (1+pct)*key[1])):
                        #2.1 if found, then append into the location
                        key_found = True
                        value.append(obj.get_text())
                        table[key] = value
                        break
                #2.2 create new location
                if not key_found:
                    table[(x0, x1)] = [obj.get_text()]
                #3.return the updated (hash)table
                return table           

                
            #1.extract text from PDF layout element 
            hash_table = {}
            figure_content = []
            for obj in objs:
                if isinstance(obj, LTTextBox) or isinstance(obj, LTTextLine):
                    #1.1 text
                    update_hash_table(hash_table, obj)
                elif isinstance(obj, LTImage):
                    #1.2 image
                    continue
                elif isinstance(obj, LTFigure):
                    #1.3 figure
                    figure_content.append(parse_lt_objs(obj))
                else:
                    #1.4 others like lines etc.
                    pass
            #2.organize the information
            text_content = self.obj_seperator.join(self.obj_seperator.join(value) for (key, value) in sorted((k, v) for (k, v) in hash_table.items()))
            return text_content
        
        
        #0.support runtime setting of file pointer
        self.fp = fp or self.fp
        if not self.fp:
            return []
            
        #1. STATE pattern
        #1.1 create document object
        instream = PDFDocument()    
        parser = PDFParser(self.fp) 
        #1.2 attach document and parser object to each other   
        parser.set_document(instream) 
        instream.set_parser(parser)
        #1.3 send message(function call) from context
        instream.initialize()     
        
        #2.create pipe structure for output
        if not instream.is_extractable:
            raise PDFTextExtractionNotAllowed     
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        #2.1 create output stream
        #outfp = file(filename+r".txt", "wb")
        #outstream = TextConverter(rsrcmgr, outfp)
        outstream = PDFPageAggregator(rsrcmgr, laparams = laparams)
        #2.2 create interpreter
        interpreter = PDFPageInterpreter(rsrcmgr, outstream)    
        
        #3.operate the interpreter and parse to get information from output
        text = []
        for raw_page in instream.get_pages():
            #3.1 raw_page |->layout
            interpreter.process_page(raw_page)
            #3.2 layout|->content by literal parse
            page = parse_lt_objs(outstream.get_result()._objs)
            #3.3 append page into text 
            text.append(page.encode("UTF-8"))
        
        return self.page_seperator.join(text) 