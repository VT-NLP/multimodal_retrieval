import json
 
import gzip
import os
import torch
from search.image_search import ImageSearcher 
from search.semantic_search import SemanticSearcher
 
 
from nltk import tokenize
from itertools import zip_longest
 
from data_util.interface.saver import Saver
from data_util.kmass_data_util import KmassSaver, load_kmass_ground_truth_data
from utils.config import config


def training_loop(args,logger,rank=0):
                
    saver=Saver()
    if args.dataset=="kmass":
        dataloader,corpus_dict=load_kmass_ground_truth_data(args.dataset_dir, args.corpus_dir)
        saver=KmassSaver(args.dataset_dir)
    if args.media=="txt":
        text_retrieve(args,corpus_dict,dataloader,saver)
    elif   args.media=="img":
        image_retrieve(args,logger,corpus_dict,dataloader,saver)
    elif args.media=="img_txt":
        text_retrieve(args,corpus_dict,dataloader,saver)
        image_retrieve(args,logger,corpus_dict,dataloader,saver)
 
    
def gen_document_text_list(corpus_dict):
    relevant_document_text_list=[]
    corpus_text_corpus_id_dict={}
    for corpus_id,entity_object in corpus_dict.items():
        relevant_document_text_list.append(entity_object.text)
        corpus_text_corpus_id_dict[entity_object.text]=corpus_id
    return relevant_document_text_list,corpus_text_corpus_id_dict

def text_retrieve(args,corpus_dict,dataloader,saver ):
    corpus_text_list,corpus_text_corpus_id_dict=gen_document_text_list(corpus_dict )
    
    searcher=SemanticSearcher(args.bi_encoder_checkpoint,args.cross_encoder_checkpoint,args.no_rerank)
    searcher.encode_corpus(corpus_text_list)
   
    valid_num=1
    precision,recall=0,0
    
    length=len(dataloader)
    for iters in range(length):
        query_id,query_text,query_image_path_list,gold_candidates =dataloader.dataset[iters]
        
        semantic_results=searcher.search(query_text,args.top_k   )
        
         
        if  iters%100==0:
            print(f"{iters}/{length}: {precision/valid_num}, {recall/valid_num}")
        if config.verbose==True:
            print(f"claim:{query_text},semantic_results:{semantic_results}")
        saver.add_retrieved_text(query_text,semantic_results,corpus_text_list,corpus_dict,corpus_text_corpus_id_dict)
    if valid_num>1:
        precision/=(valid_num-1)
        recall/=(valid_num-1)
    saver.save(args.csv_out_dir)
    print(f"{precision}, {recall},{compute_f1(precision, recall)}")
 
 
def compute_f1(precision, recall):
    if precision+recall !=0:
        f1=2*precision*recall/(precision+recall)
    else:
        f1=0
    return f1 
  
    
 
def image_retrieve(args,logger,entity_dict,dataloader ,saver):
  
    image_searcher=ImageSearcher(args.image_encoder_checkpoint,logger)
    
    if args.use_precomputed_embeddings=="y":
        use_precomputed_embeddings_flag=True 
    else:
        use_precomputed_embeddings_flag=False 
    image_searcher.encode_corpus(entity_dict ,args.image_dir,use_precomputed_embeddings_flag)
 
    precision,recall=0,0
    right_num_at_k_dict={1:0,10:0,100:0}
    retrieved_imgs_list=[]
    valid_num=1
    length=len(dataloader)
    for iters in range(length):
        id,text, img_path_list, entity_candidate_name_list  =dataloader.dataset[iters]
       
        img_path=img_path_list[0]
        if os.path.exists(img_path):
            semantic_results,retrieved_entity_name_list=image_searcher.search(img_path,args.top_k   )
        
        for k,num in right_num_at_k_dict.items():
            for entity_candidate_name  in entity_candidate_name_list:
                if entity_candidate_name in retrieved_entity_name_list[:k]:
                    right_num_at_k_dict[k]+=1
        saver.add_retrieved_text(id,img_path_list,semantic_results,retrieved_entity_name_list, entity_dict )  
            
        if iters%100==0:
            print(f"{iters}: {right_num_at_k_dict} ")
    saver.save(args.csv_out_dir)
    recall_at_k_dict={}
    for k,num in right_num_at_k_dict.items():
        
        recall_at_k_dict[k]=num/length
    acc=right_num_at_k_dict[1]/length
 
    print(f"{acc}, {recall_at_k_dict} ")

   
    

    
    
    
    
    

    
