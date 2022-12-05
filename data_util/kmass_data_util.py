
from urllib.parse import unquote
import pandas as pd
import os 
import hashlib
import re
import json
from tqdm import tqdm
import pickle

from transformers import CLIPFeatureExtractor, CLIPProcessor
  
import torch
from torch.utils.data import Dataset  
from torch.utils.data import DataLoader 
from pathlib import Path
import json
import pandas as pd

from data_util.interface.data_util import NpEncoder


class Entity:
    def __init__(self,id,image_path_list,text,title)  :
         
        self.id=id 
        self.title=title
        self.image_path_list=image_path_list
        self.text=text 
        self.entity_candidate_id_list=[]

class Query:
    def __init__(self,id,image_path_list,text,entity_candidate_name_list=[])  :
         
        self.id=id 
        self.image_path_list=image_path_list
        self.text=text 
        self.entity_candidate_id_list=entity_candidate_name_list
        
class QueryDataset(Dataset):
    def __init__(self, query_dict):
         
        self.query_dict=query_dict
        

    def __len__(self):
        return len(self.query_dict)

    def __getitem__(self, idx):
        
        keys_list = list(self.query_dict)
        key = keys_list[idx]
        news=self.query_dict[key]
        text=news.text 
        id=news.id
        image_path=news.image_path_list
         
        entity_candidate_id_list=news.entity_candidate_id_list
        
 
        return id,text, image_path, entity_candidate_id_list
 


def load_kmass_ground_truth_data(query_dir, corpus_dir,corpus_pickle_dir=None, image_dir=None):
   
    corpus_entity_dict=read_entity(corpus_dir,corpus_pickle_dir,image_dir)
    
    query_entity_dict=read_mention(query_dir )
    
    
    dataset=QueryDataset(query_entity_dict)
   
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  
    return train_dataloader,corpus_entity_dict

   
    
    
def read_mention(dataset_dir):
   
    mention_dict={}
    
    with open(dataset_dir, 'r', encoding='utf-8') as fr:
        testData = json.load(fr)
        questions=testData["questions"]
        for question  in  questions :
            query=Query(question,None,question)
            mention_dict[question]=query
           
    return mention_dict
              
    
def read_entity(corpus_path,corpus_pickle_dir,image_dir,use_cache=True ):  
    entity_dict={}
    idx=0
    document_file_list=os.listdir(corpus_path)
    for document_file in document_file_list:
        document_path=os.path.join(corpus_path,document_file)
        df_news = pd.read_csv(document_path , sep="\t",encoding="utf8")
        for _,row in tqdm(df_news.iterrows()):
            text_chunks=row['Text Chunks']
            title=row['Title']
            idx_in_csv=row['Unnamed: 0']
            id=document_file.split(".tsv")[0]+str(idx_in_csv)
            if text_chunks[0]=="'" and text_chunks[ -1]=="'":
                text_chunks=text_chunks[1:-1]
            if title[0]=="'" and title[ -1]=="'":
                title=title[1:-1]  
          
             
            entity=Entity(id,None,text_chunks ,title  )
            entity_dict[id]=entity
            idx+=1
     
    return entity_dict
        
 
import pandas as pd
import numpy as np

class KmassSaver:
    def __init__(self,dataset_dir)  :
        with open(dataset_dir, 'r', encoding='utf-8') as fr:
            testData = json.load(fr)
            task_id=testData["id"]
        answer_dir="data/kmass/queryio/out/merged__question.json"
        with open(answer_dir, 'r', encoding='utf-8') as fr:
            answerData = json.load(fr)
            output=answerData["output"]
            answer_dict={}
            for one_item in output:
                question=one_item["question"]
                answer=one_item["answer"]
                answer_ctx=one_item["answer_ctx"]
                answer_dict[question]={"answer":answer,"answer_ctx":answer_ctx}
            self.answer_dict=answer_dict
        self.json_object={"id":task_id,"config": None,"output":[]}
         

    def add_retrieved_text(self,query ,semantic_results,corpus_text_list,corpus_dict,corpus_text_corpus_id_dict):
        answer_dict=self.answer_dict
        answer_json=answer_dict[query]
        answer=answer_json["answer"]
        answer_ctx=answer_json["answer_ctx"]
        
        one_question_json={'question':query,"ctxs":[]}
        for hit in semantic_results:
            corpus_text=corpus_text_list[hit['corpus_id']]
            corpus_id=corpus_text_corpus_id_dict[corpus_text]
            entity=corpus_dict[corpus_id]
             
            one_question_json["ctxs"].append({
                "id":entity.id,"title":entity.title,"text":entity.text,"score":hit["cross-score"] ,"bi_encoder_score":hit["score"]*100,
                "has_answer":has_answer(answer,entity.text)
            })
        
        one_question_json["answer"]=answer
        one_question_json["answer_ctx"]=answer_ctx
        self.json_object["output"].append(one_question_json)

    def save(self,out_path):
        with open(out_path, 'w', encoding='utf-8') as fp:
            json.dump(self.json_object, fp, indent=4, cls=NpEncoder)  
       

def has_answer(answer,retrieved_text):
    if answer.lower() in retrieved_text.lower():
        return True 
    else:
        return False
     
        
def check_precision(dataset_dir):
    precision_recall_at_k=[1, 3, 5, 10,100]
    precisions_at_k = {k: [] for k in  precision_recall_at_k}
    recall_at_k = {k: [] for k in  precision_recall_at_k}
    
    
    with open(dataset_dir, 'r', encoding='utf-8') as fr:
        testData = json.load(fr)
        output=testData["output"]
        for one_item in output:
            answer_candidates=one_item["ctxs"]
    
            for k_val in  precision_recall_at_k:
                num_correct = 0
                for hit in answer_candidates[0:k_val]:
                    if hit['has_answer']:
                        num_correct += 1

                precisions_at_k[k_val].append(num_correct / k_val)
                
        for k in precisions_at_k:
            precisions_at_k[k] = round(np.mean(precisions_at_k[k])*100,2)

 
    print(precisions_at_k)
    

def check_precision_based_on_one_ground_truth(dataset_dir):
    precision_recall_at_k=[1, 3, 5, 10,100]
    precisions_at_k = {k: [] for k in  precision_recall_at_k}
    recall_at_k = {k: [] for k in  precision_recall_at_k}
    
    
    with open(dataset_dir, 'r', encoding='utf-8') as fr:
        testData = json.load(fr)
        output=testData["output"]
        for one_item in output:
            answer_candidates=one_item["ctxs"]
            if one_item["answer_ctx"] is not None:
                gold_entity_id=one_item["answer_ctx"]["id"]
                for k_val in  precision_recall_at_k:
                    num_correct = 0
                    for hit in answer_candidates[0:k_val]:
                        
                        if hit['id']==gold_entity_id:
                            num_correct += 1

                    precisions_at_k[k_val].append(num_correct / k_val)
                 
        for k in precisions_at_k:
            precisions_at_k[k] = round(np.mean(precisions_at_k[k])*100,2)

         
    print(precisions_at_k)
    
    
if __name__=="__main__":
    # dataset_dir="data/kmass/queryio/out/merged__question.json"
    dataset_dir="data/kmass/queryio/out/merged__question_all-mpnet-base-v4.json"
    check_precision_based_on_one_ground_truth(dataset_dir)
    
    
    