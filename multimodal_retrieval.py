import os
import click
import re
import json
import tempfile
import torch
from utils.util import setup
 
from training import training_loop
 
 
import numpy as np
import logging 
logging.basicConfig(filename="log.txt",
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
#----------------------------------------------------------------------------



@click.command()
@click.pass_context
@click.option('--dataset', type=str,default="kmass" )
@click.option('--dataset_dir', help='input: query', required=True, metavar='DIR',default="data/kmass/queryio/in/question.json")
@click.option('--corpus_dir', help='input: document', required=True, metavar='DIR',default='data/kmass/documents' )
@click.option('--image_dir', help='optional input: image folder',   metavar='DIR' ) 
@click.option('--corpus_pickle_dir', help='optional input: saved corpus embedding pickle', metavar='DIR') 
@click.option('--csv_out_dir', help='Where to save the output file',default="data/kmass/queryio/out/merged__question_all-mpnet-base-v4.json",   metavar='DIR' )
@click.option('--outdir', help='Where to save the log', required=True, metavar='DIR',default="output/runs")
@click.option('--top_k', help='top_k', type=int,default=100, metavar='INT')  
@click.option('--media', type=str,default="txt" ) #txt,img_txt
@click.option('--use_precomputed_embeddings',  help="use saved corpus embedding pickle from $corpus_pickle_dir$",type=str,default="y" )  
@click.option('--bi_encoder_checkpoint',  metavar='DIR',default="all-mpnet-base-v2")
@click.option('--cross_encoder_checkpoint',  metavar='DIR',default="cross-encoder/ms-marco-MiniLM-L-12-v2")
@click.option('--image_encoder_checkpoint',  metavar='DIR',default="clip-ViT-L-14")  
@click.option('--no_rerank', help='remove the reranker from text retrieval pipeline', is_flag=True, show_default=True, default=False   )  
def main(ctx,  **config_kwargs):
    args,logger=setup(config_kwargs)
    logging.info("start retrieval")
    training_loop.training_loop(args,logger,rank=0)
  

if __name__ == "__main__":
    
    main() 