from __future__ import print_function

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as K
import sys
import random
import traceback
import pickle
from keras.optimizers import RMSprop, Adam,SGD
from scipy.stats import rankdata
import math
from math import log
from models import *
import argparse
random.seed(42)
import threading
import tables  
import configs
import codecs
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

from utils import cos_np, normalize, cos_np_for_normalized
from configs import get_config
from models import CARLCS_CNN



class CodeSearcher:
    def __init__(self, conf=None):
        self.conf = dict() if conf is None else conf
        self.path = self.conf.get('workdir', '../data/github/')
        self.train_params = conf.get('training_params', dict())
        self.data_params=conf.get('data_params',dict())
        self.model_params=conf.get('model_params',dict())
                
        self.vocab_methname = self.load_pickle(self.path+self.data_params['vocab_methname'])
        self.vocab_apiseq=self.load_pickle(self.path+self.data_params['vocab_apiseq'])
        self.vocab_tokens=self.load_pickle(self.path+self.data_params['vocab_tokens'])
        self.vocab_desc=self.load_pickle(self.path+self.data_params['vocab_desc'])
        
        self._eval_sets = None

    def load_pickle(self, filename):
        return pickle.load(open(filename, 'rb'))    

    ##### Data Set #####
    def load_training_data_chunk(self,offset,chunk_size):
        logger.debug('Loading a chunk of training data..')
        logger.debug('methname')
        chunk_methnames=self.load_hdf5(self.path+self.data_params['train_methname'],offset,chunk_size)
        logger.debug('apiseq')
        chunk_apiseqs=self.load_hdf5(self.path+self.data_params['train_apiseq'],offset,chunk_size)
        logger.debug('tokens')
        chunk_tokens=self.load_hdf5(self.path+self.data_params['train_tokens'],offset,chunk_size)
        logger.debug('desc')
        chunk_descs=self.load_hdf5(self.path+self.data_params['train_desc'],offset,chunk_size)   
        return chunk_methnames,chunk_apiseqs,chunk_tokens,chunk_descs    
    def load_valid_data_chunk(self,chunk_size):
        logger.debug('Loading a chunk of validation data..')
        logger.debug('methname')
        chunk_methnames=self.load_hdf5(self.path+self.data_params['valid_methname'],0,chunk_size)
        logger.debug('apiseq')
        chunk_apiseqs=self.load_hdf5(self.path+self.data_params['valid_apiseq'],0,chunk_size)
        logger.debug('tokens')
        chunk_tokens=self.load_hdf5(self.path+self.data_params['valid_tokens'],0,chunk_size)
        logger.debug('desc')
        chunk_descs=self.load_hdf5(self.path+self.data_params['valid_desc'],0,chunk_size)   
        return chunk_methnames,chunk_apiseqs,chunk_tokens,chunk_descs   
    def load_hdf5(self,vecfile,start_offset,chunk_size):
        """reads training sentences(list of int array) from a hdf5 file"""  
        table = tables.open_file(vecfile)
        data, index = (table.get_node('/phrases'),table.get_node('/indices'))
        data_len = index.shape[0]
        if chunk_size==-1:#if chunk_size is set to -1, then, load all data
            chunk_size=data_len
        start_offset = start_offset%data_len
        offset=start_offset
        logger.debug("{} entries".format(data_len))
        logger.debug("starting from offset {} to {}".format(start_offset,start_offset+chunk_size))
        sents = []
        while offset < start_offset+chunk_size:
            if offset>=data_len:   
                logger.warn('Warning: offset exceeds data length, starting from index 0..')             
                chunk_size=start_offset+chunk_size-data_len
                start_offset=0
                offset=0
            len, pos = index[offset]['length'], index[offset]['pos']
            offset += 1
            sents.append(data[pos:pos + len].astype('int32'))
        table.close()
        return sents 
    
    
    ##### Converting / reverting #####
    def convert(self, vocab, words):
        """convert words into indices"""       
        if type(words) == str:
            words = words.strip().lower().split(' ')
        return [vocab.get(w, 0) for w in words]
    def revert(self, vocab, indices):
        """revert indices into words"""
        ivocab = dict((v, k) for k, v in vocab.items())
        return [ivocab.get(i, 'UNK') for i in indices]

    ##### Padding #####
    def pad(self, data, len=None):
        from keras.preprocessing.sequence import pad_sequences
        return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)
    
       
       
    ##### Model Loading / saving #####
    def save_model_epoch(self, model, epoch):
        if not os.path.exists(self.path+'models/'+self.model_params['model_name']+'/'):
            os.makedirs(self.path+'models/'+self.model_params['model_name']+'/')
        model.save("{}models/{}/epo{:d}_code.h5".format(self.path, self.model_params['model_name'], epoch),
               "{}models/{}/epo{:d}_desc.h5".format(self.path, self.model_params['model_name'], epoch), overwrite=True)
        
    def load_model_epoch(self, model, epoch):
        assert os.path.exists(
            "{}models/{}/epo{:d}_code.h5".format(self.path, self.model_params['model_name'], epoch))\
            ,"Weights at epoch {:d} not found".format(epoch)
        assert os.path.exists(
            "{}models/{}/epo{:d}_desc.h5".format(self.path, self.model_params['model_name'], epoch))\
            ,"Weights at epoch {:d} not found".format(epoch)
        model.load("{}models/{}/epo{:d}_code.h5".format(self.path, self.model_params['model_name'], epoch),
               "{}models/{}/epo{:d}_desc.h5".format(self.path, self.model_params['model_name'], epoch))



    ##### Training #####
    def train(self, model):
        if self.train_params['reload']>0:
            self.load_model_epoch(model, self.train_params['reload'])
        valid_every = self.train_params.get('valid_every', None)
        save_every = self.train_params.get('save_every', None)
        batch_size = self.train_params.get('batch_size', 128)
        nb_epoch = self.train_params.get('nb_epoch', 10)
        split = self.train_params.get('validation_split', 0)
        
        val_loss = {'loss': 1., 'epoch': 0}
        for i in range(self.train_params['reload']+1, nb_epoch):
            print('Epoch %d :: \n' % i, end='')            
            logger.debug('loading data chunk..')
            chunk_methnames,chunk_apiseqs,chunk_tokens,chunk_descs =\
                    self.load_training_data_chunk(\
                                        (i-1)*self.train_params.get('chunk_size', 100000),\
                                        self.train_params.get('chunk_size', 100000))
            logger.debug('padding data..')
            chunk_padded_methnames = self.pad(chunk_methnames, self.data_params['methname_len'])
            chunk_padded_apiseqs = self.pad(chunk_apiseqs, self.data_params['apiseq_len'])
            chunk_padded_tokens = self.pad(chunk_tokens, self.data_params['tokens_len'])
            chunk_padded_good_descs = self.pad(chunk_descs,self.data_params['desc_len'])
            chunk_bad_descs=[desc for desc in chunk_descs]
            random.shuffle(chunk_bad_descs)
            chunk_padded_bad_descs = self.pad(chunk_bad_descs, self.data_params['desc_len'])
            hist = model.fit([chunk_padded_methnames,chunk_padded_apiseqs,chunk_padded_tokens, chunk_padded_good_descs, chunk_padded_bad_descs], epochs=1, batch_size=batch_size, validation_split=split)

            if hist.history['val_loss'][0] < val_loss['loss']:
                val_loss = {'loss': hist.history['val_loss'][0], 'epoch': i}
            print('Best: Loss = {}, Epoch = {}'.format(val_loss['loss'], val_loss['epoch']))

            if valid_every is not None and i % valid_every == 0:
                acc1,mrr = self.valid(model,1000,1)
                        
            if save_every is not None and i % save_every == 0:
                self.save_model_epoch(model, i)
    
    def valid(self, model, poolsize,K):
        """
        quick validation in a code pool. 
        param:
            poolsize - size of the code pool, if -1, load the whole test set
        """
        #load test dataset
        if self._eval_sets is None:
            #self._eval_sets = dict([(s, self.load(s)) for s in ['dev', 'test1', 'test2']])
            methnames,apiseqs,tokens,descs=self.load_valid_data_chunk(poolsize)
            self._eval_sets=dict()
            self._eval_sets['methnames']=methnames
            self._eval_sets['apiseqs']=apiseqs
            self._eval_sets['tokens']=tokens
            self._eval_sets['descs']=descs
           
        c_1, c_2 = 0, 0        
        data_len=len(self._eval_sets['descs'])
        for i in range(data_len):
            bad_descs=[desc for desc in self._eval_sets['descs']]
            random.shuffle(bad_descs)
            descs=bad_descs 
            descs[0]=self._eval_sets['descs'][i]#good desc
            descs=self.pad(descs,self.data_params['desc_len'])
            methnames=self.pad([self._eval_sets['methnames'][i]]*data_len,self.data_params['methname_len'])
            apiseqs=self.pad([self._eval_sets['apiseqs'][i]]*data_len,self.data_params['apiseq_len'])
            tokens=self.pad([self._eval_sets['tokens'][i]]*data_len,self.data_params['tokens_len'])
            n_good = K
            
            sims = model.predict([methnames, apiseqs,tokens, descs], batch_size=data_len).flatten()
            r = rankdata(sims, method='max')
            max_r = np.argmax(r)
            max_n = np.argmax(r[:n_good])
            c_1 += 1 if max_r == max_n else 0
            c_2 += 1 / float(r[max_r] - r[max_n] + 1)

        top1 = c_1 / float(data_len)
            #percentage of predicted most similar desc that is really the corresponding desc
        mrr = c_2 / float(data_len)
        logger.info('Top-1 Precision={}, MRR={}'.format(top1,mrr))
        
        return top1, mrr  
    

    ##### Evaluation in the develop set #####
    def eval(self, model, poolsize, K):
        """
        evaluate in a evaluation date. 
        param:
            poolsize - size of the code pool, if -1, load the whole test set
        """
        def Recall(real,predict,n_results):
            sum=0.0
            for val in real:
                try: index=predict.index(val)
                except ValueError: index=-1
                if index <= n_results: sum=sum+1  
            return sum/float(len(real))
        def ACC(real,predict):
            sum=0.0
            for val in real:
                try: index=predict.index(val)
                except ValueError: index=-1
                if index!=-1: sum=sum+1  
            return sum/float(len(real))
        def MAP(real,predict):
            sum=0.0
            for id,val in enumerate(real):
                try: index=predict.index(val)
                except ValueError: index=-1
                if index!=-1: sum=sum+(id+1)/float(index+1)
            return sum/float(len(real))
        def MRR(real,predict):
            sum=0.0
            for val in real:
                try: index=predict.index(val)
                except ValueError: index=-1
                if index!=-1: sum=sum+1.0/float(index+1)
            return sum/float(len(real))
        def NDCG(real,predict):
            dcg=0.0
            idcg=IDCG(len(real))
            for i,predictItem in enumerate(predict):
                if predictItem in real:
                    itemRelevance=1
                    rank = i+1
                    dcg+=(math.pow(2,itemRelevance)-1.0)*(math.log(2)/math.log(rank+1))
            return dcg/float(idcg)
        def IDCG(n):
            idcg=0
            itemRelevance=1
            for i in range(n):
                idcg+=(math.pow(2,itemRelevance)-1.0)*(math.log(2)/math.log(i+2))
            return idcg

        #load valid dataset
        if self._eval_sets is None:
            methnames,apiseqs,tokens,descs=self.load_valid_data_chunk(poolsize)
            self._eval_sets=dict()
            self._eval_sets['methnames']=methnames
            self._eval_sets['apiseqs']=apiseqs
            self._eval_sets['tokens']=tokens
            self._eval_sets['descs']=descs
        recall,acc,mrr,map,ndcg=0,0,0,0,0
        data_len=len(self._eval_sets['descs'])
        for i in range(data_len):
            print(i) 
            desc=self._eval_sets['descs'][i]#good desc
            descs=self.pad([desc]*data_len,self.data_params['desc_len'])
            methnames=self.pad(self._eval_sets['methnames'],self.data_params['methname_len'])
            apiseqs=self.pad(self._eval_sets['apiseqs'],self.data_params['apiseq_len'])
            tokens=self.pad(self._eval_sets['tokens'],self.data_params['tokens_len'])
            n_results = K          
            sims = model.predict([methnames,apiseqs,tokens, descs], batch_size=data_len).flatten()
            negsims=np.negative(sims)
            predict_origin=np.argsort(negsims)#predict = np.argpartition(negsims, kth=n_results-1)
            predict = predict_origin[:n_results]   
            predict = [int(k) for k in predict]
            predict_origin = [int(k) for k in predict_origin]
            real=[i]
            recall+=Recall(real,predict_origin,n_results)
            acc+=ACC(real,predict)
            mrr+=MRR(real,predict)
            map+=MAP(real,predict)
            ndcg+=NDCG(real,predict)
        recall = recall / float(data_len)                        
        acc = acc / float(data_len)
        mrr = mrr / float(data_len)
        map = map / float(data_len)
        ndcg= ndcg/ float(data_len)
        logger.info('Recall={}, ACC={}, MRR={}, MAP={}, nDCG={}'.format(recall,acc,mrr,map,ndcg))
        return recall,acc,mrr,map,ndcg



    
def parse_args():
    parser = argparse.ArgumentParser("Train and Test Model")
    parser.add_argument("--proto", choices=["get_config"],  default="get_config",
                        help="Prototype config to use for config")
    parser.add_argument("--mode", choices=["train","eval"], default='train',
                        help="The mode to run. The `train` mode trains a model;"
                        " the `eval` mode evaluat models in a test set ")
    parser.add_argument("--verbose",action="store_true", default=True, help="Be verbose")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    conf = getattr(configs, args.proto)()
    codesearcher = CodeSearcher(conf)

    ##### Define model ######
    logger.info('Build Model')
    model = eval(conf['model_params']['model_name'])(conf)#initialize the model
    model.build()
    optimizer=Adam(clipnorm=0.1)
    model.compile(optimizer=optimizer)
    
    if args.mode=='train':  
        codesearcher.train(model)
        
    elif args.mode=='eval':
        # evaluation
        #load model
        if conf['training_params']['reload']>0:
            codesearcher.load_model_epoch(model, conf['training_params']['reload'])
        codesearcher.eval(model,-1,10)



