import collections
import pickle
from dis import dis
import os
import random
from pathlib import Path
import logging
import shutil
from packaging import version

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
from param import parse_args
from pretrain_data import get_loader,load_pickle 
from utils import LossMeter
from dist_utils import reduce_dict, new_reduce_dict

from peft import (    # LoRA Setting
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

_use_native_amp = False
_use_apex = False

_use_native_amp = True
from torch.cuda.amp import autocast

from trainer_base import TrainerBase
from pretrain_model import InstructGLM

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

class FP(nn.Module):    # first_model, i.e. simple MLP for dimension transformation of OGB/ GIANT node embedding + freezed Llama-7b word embeddings.
    def __init__(self,llama_embed,real):
        super(FP,self).__init__()
        self.trans_2=nn.Linear(512,4096,bias=False)
        self.trans_1=nn.Linear(128,512,bias=False)
        self.rac=nn.ELU()
        self.sln=nn.LayerNorm(512)

        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
            elif isinstance(m,nn.Linear):
                m.weight.data.normal_(mean=0.0, std=1.0)

        self.embed_tokens=llama_embed   # freezed Llama-7b word embeddings.

        self.real=real
        

    def forward(self, input_ids):
        transfered=self.trans_2(self.rac(self.sln(self.trans_1(self.real)))) 

        inputs_embeds = transfered[input_ids] + self.embed_tokens[input_ids] ### embedding step - add HERE ###
        # transfered contains shaped OGB/ Giant node embedding, while self.embed_tokens contains natural language word embedding.

        return inputs_embeds


# The Trainer inherits TrainerBase in trainer_base.py
class Trainer(TrainerBase):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True,val_list=None):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train) 


        model_class = InstructGLM
        self.m_class = InstructGLM     

        config = self.create_config()
        self.m_config=config    
        self.tokenizer = self.create_tokenizer()

        re_start=0  # main model restart way control
        if train:  

            self.model = self.create_model(model_class, config)
            self.model.tokenizer = self.tokenizer   

            self.model = prepare_model_for_int8_training(self.model)

            # Form peft-model.
            lora_r=16 
            lora_alpha=16
            lora_target_modules=['q_proj','k_proj','v_proj','o_proj','lm_head']  # Select LoRA tuning modules.
            lora_dropout=0.05

            LORA_config = LoraConfig(r=lora_r,lora_alpha=lora_alpha,target_modules=lora_target_modules,lora_dropout=lora_dropout,bias="none",task_type="CAUSAL_LM")
            if re_start!=2:
                self.model = get_peft_model(self.model, LORA_config)

            if self.verbose and re_start!=2:
                print()
                self.model.print_trainable_parameters()   
                print()
            dist.barrier()
        
            if re_start==1:
                print('Main model re-starting')
                doc_prefix='./your_folder_name_with_pickle/'
                for gg in range(32):
                    self.model.base_model.model.model.layers[gg].self_attn.q_proj.lora_A.default.weight.data=load_pickle(doc_prefix+"Gqa_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                    self.model.base_model.model.model.layers[gg].self_attn.k_proj.lora_A.default.weight.data=load_pickle(doc_prefix+"Gka_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                    self.model.base_model.model.model.layers[gg].self_attn.v_proj.lora_A.default.weight.data=load_pickle(doc_prefix+"Gva_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                    self.model.base_model.model.model.layers[gg].self_attn.o_proj.lora_A.default.weight.data=load_pickle(doc_prefix+"Goa_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                    self.model.base_model.model.model.layers[gg].self_attn.q_proj.lora_B.default.weight.data=load_pickle(doc_prefix+"Gqb_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                    self.model.base_model.model.model.layers[gg].self_attn.k_proj.lora_B.default.weight.data=load_pickle(doc_prefix+"Gkb_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                    self.model.base_model.model.model.layers[gg].self_attn.v_proj.lora_B.default.weight.data=load_pickle(doc_prefix+"Gvb_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                    self.model.base_model.model.model.layers[gg].self_attn.o_proj.lora_B.default.weight.data=load_pickle(doc_prefix+"Gob_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                self.model.base_model.model.lm_head.lora_A.default.weight.data=load_pickle(doc_prefix+"Glm_a_{}.pkl".format(self.args.lr)).data.to(args.gpu)
                self.model.base_model.model.lm_head.lora_B.default.weight.data=load_pickle(doc_prefix+"Glm_b_{}.pkl".format(self.args.lr)).data.to(args.gpu)
                print('Main model loaded.')
                
                if self.verbose:
                    self.model.save_pretrained("G_restart")  # Another way for checkpoint saving
                dist.barrier()

                for n,p in self.model.named_parameters():
                    if 'lora' in n:
                        p.requires_grad_()
                if self.verbose:
                    print()
                    self.model.print_trainable_parameters()
                    print()

            if re_start==2:    # recommended
                from peft import PeftModel, PeftConfig
                peft_model_id ='./your_folder_name_with_bin'
                print('now we are loading peft model')

                self.model = PeftModel.from_pretrained(self.model, peft_model_id)   # Another way for loading checkpoint
                for n,p in self.model.named_parameters():
                    if 'lora' in n:
                        p.requires_grad_()
                if self.verbose:
                    print()
                    self.model.print_trainable_parameters()
                    print()

            self.model = self.model.to(args.gpu)
        else:   # Inference
            self.model = self.create_model(model_class, config)
            self.model.tokenizer = self.tokenizer   

            for name, param in self.model.named_parameters():
                 # freeze base model's layers
                param.requires_grad = False

                 # cast all non INT8 parameters to fp32
            for param in self.model.parameters():
                if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
                    param.data = param.data.to(torch.float32)

            # Form peft-model
            lora_r=16 
            lora_alpha=16
            lora_target_modules=['q_proj','k_proj','v_proj','o_proj','lm_head']
            lora_dropout=0.05

            LORA_config = LoraConfig(r=lora_r,lora_alpha=lora_alpha,target_modules=lora_target_modules,lora_dropout=lora_dropout,bias="none",task_type="CAUSAL_LM")
            self.model = get_peft_model(self.model, LORA_config)

            if self.verbose:
                print()
                self.model.print_trainable_parameters()
                print()
            dist.barrier()
            self.model = self.model.to(args.gpu)
           
           # Set up first_model
        if train:
            self.first_model=FP(llama_embed=self.train_loader.dataset.llama_embed.to(args.gpu),real=self.train_loader.dataset.real_feature.to(args.gpu))
        else:
            self.first_model=FP(llama_embed=self.val_loader.dataset.llama_embed.to(args.gpu),real=self.val_loader.dataset.real_feature.to(args.gpu))


        re_start=0  # first_model restart way contral
        if train and re_start==1:
            print('All processes re-starting first-model')
            ckpt_path="yours.pth"
        
            self.load_checkpoint(ckpt_path)
            if self.verbose:
                print('first_model loaded.')
        if train and re_start==2:  
            self.first_model.sln.bias.data=load_pickle('./first_restart/sln_bias.pkl')
            self.first_model.sln.weight.data=load_pickle('./first_restart/sln_weight.pkl')
            self.first_model.trans_1.weight.data=load_pickle('./first_restart/trans_1_weight.pkl')
            self.first_model.trans_2.weight.data=load_pickle('./first_restart/trans_2_weight.pkl')
            print('first model loaded')

        self.first_model=self.first_model.to(args.gpu)
        
    

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()

        if args.multiGPU and not args.inference:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()    

            if args.distributed:  # DDP setup
                self.model = DDP(self.model, device_ids=[args.gpu])
                self.first_model=DDP(self.first_model, device_ids=[args.gpu])
                
        if self.verbose:
            print(f'It took {time() - start:.1f}s')

        self.val_list=val_list


    def train(self):
        LOSSES_NAME = self.args.LOSSES_NAME

        if self.verbose:
            loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]
            best_eval_loss = 100000.

            project_name = "Natural Language is All a Graph Needs"

            src_dir = Path(__file__).resolve().parent
            base_path = str(src_dir.parent)
            src_dir = str(src_dir)

        if self.args.distributed:
            dist.barrier()

        for epoch in range(self.args.epoch):
            global_step=0
        

            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)   

            # Train
            self.model.train()     
            self.first_model.train()

            if self.verbose:
                pbar = tqdm(total=len(self.train_loader), ncols=275)

            epoch_results = {}   
            for loss_name in LOSSES_NAME:
                epoch_results[loss_name] = 0.
                epoch_results[f'{loss_name}_count'] = 0

            for step_i, batch in enumerate(self.train_loader):  
                torch.cuda.empty_cache()
                dist.barrier()

                if self.args.fp16 and _use_native_amp:
                    pass
                else:
                    if self.args.distributed:
                        
                        dddd = next(self.model.parameters()).device

                        input_ids = batch['input_ids'].to(dddd)
                        lm_labels = batch["target_ids"].to(dddd)
                        attention_mask=batch['attn_mask'].to(dddd)

                        loss_weights = batch["loss_weights"].to(dddd)
                        B, L = lm_labels.size()

                        embeds = self.first_model(  #forward
                            input_ids=input_ids
                        )
                        #forward
                        # Notably, our input for self.mode here is numberical inputs_embeds, 
                        # i.e. we already map inputs_ids to embeddings in via self.first_model
                        output=self.model(inputs_embeds=embeds,attention_mask=attention_mask,labels=lm_labels)

                        lm_mask = lm_labels[:,1:] != -100
                        lm_mask = lm_mask.float()

                        loss = output['loss']

                        loss = loss.view(B, L-1) * lm_mask   

                        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)   

                        results = {}     

                        results['loss'] = (loss * loss_weights).mean()    
                        results['total_loss'] = loss.detach().sum()
                        results['total_loss_count'] = len(loss)

                        task_counts = {task: 0 for task in self.model.module.losses}
                        task_loss = {task: 0 for task in self.model.module.losses}

                        for _loss, task in zip(loss.detach(), batch['task']):
                            task_loss[task] += _loss
                            task_counts[task] += 1

                        for task in self.model.module.losses:
                            if task_counts[task] > 0:
                                results[f'{task}_loss'] = task_loss[task]
                                results[f'{task}_loss_count'] = task_counts[task]

                    else:
                        results = self.model.train_step(batch)

                loss = results['loss']/self.args.gradient_accumulation_steps
                dist.barrier()
                torch.cuda.empty_cache()
                
                loss.backward()

                loss = loss.detach()

                # Update Parameters
                if step_i % self.args.gradient_accumulation_steps==0:
                    
                    if self.args.clip_grad_norm > 0:
                        if self.args.fp16 and _use_native_amp:
                            self.scaler.unscale_(self.optim)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                        elif self.args.fp16 and _use_apex:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optim), self.args.clip_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                            torch.nn.utils.clip_grad_norm_(self.first_model.parameters(), self.args.clip_grad_norm)

                    if self.args.fp16 and _use_native_amp:
                        self.scaler.step(self.optim)
                        self.scaler.update()
                    else:
                        self.optim.step()    #update

                    if self.lr_scheduler:
                        self.lr_scheduler.step()
 
                    for param in self.model.parameters():    
                        param.grad = None
                    for param in self.first_model.parameters():    
                        param.grad = None

                global_step += 1
                
                # 32 attention block layers in 7b 
                # Another way for saving checkpoint is 'self.model.save_pretrained("checkpoint_fold_name")'
                if global_step==len(self.train_loader)//8:
                    if self.verbose:
                        for ig in range(32):   
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_A.default.weight.data,"./llama_{}_mmid1/Gqa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_A.default.weight.data,"./llama_{}_mmid1/Gka_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_A.default.weight.data,"./llama_{}_mmid1/Gva_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_A.default.weight.data,"./llama_{}_mmid1/Goa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_B.default.weight.data,"./llama_{}_mmid1/Gqb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_B.default.weight.data,"./llama_{}_mmid1/Gkb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_B.default.weight.data,"./llama_{}_mmid1/Gvb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_B.default.weight.data,"./llama_{}_mmid1/Gob_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_A.default.weight,"./llama_{}_mmid1/Glm_a_{}.pkl".format(epoch+1,self.args.lr))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_B.default.weight,"./llama_{}_mmid1/Glm_b_{}.pkl".format(epoch+1,self.args.lr))

                        torch.save(self.first_model.state_dict(),"Gfirst_{}_{}_8_{}_mmid1.pth".format(epoch+1,self.args.lr,self.args.train))
                if global_step==len(self.train_loader)//4:
                    if self.verbose:
                        for ig in range(32):
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_A.default.weight.data,"./llama_{}_mid1/Gqa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_A.default.weight.data,"./llama_{}_mid1/Gka_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_A.default.weight.data,"./llama_{}_mid1/Gva_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_A.default.weight.data,"./llama_{}_mid1/Goa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_B.default.weight.data,"./llama_{}_mid1/Gqb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_B.default.weight.data,"./llama_{}_mid1/Gkb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_B.default.weight.data,"./llama_{}_mid1/Gvb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_B.default.weight.data,"./llama_{}_mid1/Gob_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_A.default.weight,"./llama_{}_mid1/Glm_a_{}.pkl".format(epoch+1,self.args.lr))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_B.default.weight,"./llama_{}_mid1/Glm_b_{}.pkl".format(epoch+1,self.args.lr))

                        torch.save(self.first_model.state_dict(),"Gfirst_{}_{}_8_{}_mid1.pth".format(epoch+1,self.args.lr,self.args.train))
                if global_step==len(self.train_loader)*3//8:
                    if self.verbose:
                        for ig in range(32):
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_A.default.weight.data,"./llama_{}_mmid2/Gqa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_A.default.weight.data,"./llama_{}_mmid2/Gka_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_A.default.weight.data,"./llama_{}_mmid2/Gva_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_A.default.weight.data,"./llama_{}_mmid2/Goa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_B.default.weight.data,"./llama_{}_mmid2/Gqb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_B.default.weight.data,"./llama_{}_mmid2/Gkb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_B.default.weight.data,"./llama_{}_mmid2/Gvb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_B.default.weight.data,"./llama_{}_mmid2/Gob_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_A.default.weight,"./llama_{}_mmid2/Glm_a_{}.pkl".format(epoch+1,self.args.lr))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_B.default.weight,"./llama_{}_mmid2/Glm_b_{}.pkl".format(epoch+1,self.args.lr))

                        torch.save(self.first_model.state_dict(),"Gfirst_{}_{}_8_{}_mmid2.pth".format(epoch+1,self.args.lr,self.args.train))
                if global_step==len(self.train_loader)//2:
                    if self.verbose:
                        for ig in range(32):
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_A.default.weight.data,"./llama_{}_mid2/Gqa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_A.default.weight.data,"./llama_{}_mid2/Gka_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_A.default.weight.data,"./llama_{}_mid2/Gva_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_A.default.weight.data,"./llama_{}_mid2/Goa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_B.default.weight.data,"./llama_{}_mid2/Gqb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_B.default.weight.data,"./llama_{}_mid2/Gkb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_B.default.weight.data,"./llama_{}_mid2/Gvb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_B.default.weight.data,"./llama_{}_mid2/Gob_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_A.default.weight,"./llama_{}_mid2/Glm_a_{}.pkl".format(epoch+1,self.args.lr))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_B.default.weight,"./llama_{}_mid2/Glm_b_{}.pkl".format(epoch+1,self.args.lr))

                        torch.save(self.first_model.state_dict(),"Gfirst_{}_{}_8_{}_mid2.pth".format(epoch+1,self.args.lr,self.args.train))
                if global_step==len(self.train_loader)*5//8:
                    if self.verbose:
                        for ig in range(32):
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_A.default.weight.data,"./llama_{}_mmid3/Gqa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_A.default.weight.data,"./llama_{}_mmid3/Gka_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_A.default.weight.data,"./llama_{}_mmid3/Gva_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_A.default.weight.data,"./llama_{}_mmid3/Goa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_B.default.weight.data,"./llama_{}_mmid3/Gqb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_B.default.weight.data,"./llama_{}_mmid3/Gkb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_B.default.weight.data,"./llama_{}_mmid3/Gvb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_B.default.weight.data,"./llama_{}_mmid3/Gob_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_A.default.weight,"./llama_{}_mmid3/Glm_a_{}.pkl".format(epoch+1,self.args.lr))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_B.default.weight,"./llama_{}_mmid3/Glm_b_{}.pkl".format(epoch+1,self.args.lr))

                        torch.save(self.first_model.state_dict(),"Gfirst_{}_{}_8_{}_mmid3.pth".format(epoch+1,self.args.lr,self.args.train))
                if global_step==len(self.train_loader)*3//4:
                    if self.verbose:
                        for ig in range(32):
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_A.default.weight.data,"./llama_{}_mid3/Gqa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_A.default.weight.data,"./llama_{}_mid3/Gka_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_A.default.weight.data,"./llama_{}_mid3/Gva_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_A.default.weight.data,"./llama_{}_mid3/Goa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_B.default.weight.data,"./llama_{}_mid3/Gqb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_B.default.weight.data,"./llama_{}_mid3/Gkb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_B.default.weight.data,"./llama_{}_mid3/Gvb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_B.default.weight.data,"./llama_{}_mid3/Gob_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_A.default.weight,"./llama_{}_mid3/Glm_a_{}.pkl".format(epoch+1,self.args.lr))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_B.default.weight,"./llama_{}_mid3/Glm_b_{}.pkl".format(epoch+1,self.args.lr))

                        torch.save(self.first_model.state_dict(),"Gfirst_{}_{}_8_{}_mid3.pth".format(epoch+1,self.args.lr,self.args.train))
                if global_step==len(self.train_loader)*7//8:
                    if self.verbose:
                        for ig in range(32):
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_A.default.weight.data,"./llama_{}_mend/Gqa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_A.default.weight.data,"./llama_{}_mend/Gka_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_A.default.weight.data,"./llama_{}_mend/Gva_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_A.default.weight.data,"./llama_{}_mend/Goa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_B.default.weight.data,"./llama_{}_mend/Gqb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_B.default.weight.data,"./llama_{}_mend/Gkb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_B.default.weight.data,"./llama_{}_mend/Gvb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                            save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_B.default.weight.data,"./llama_{}_mend/Gob_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_A.default.weight,"./llama_{}_mend/Glm_a_{}.pkl".format(epoch+1,self.args.lr))
                        save_pickle(self.model.module.base_model.model.lm_head.lora_B.default.weight,"./llama_{}_mend/Glm_b_{}.pkl".format(epoch+1,self.args.lr))

                        torch.save(self.first_model.state_dict(),"Gfirst_{}_{}_8_{}_mend.pth".format(epoch+1,self.args.lr,self.args.train))

                dist.barrier()

                if self.lr_scheduler:
                    if version.parse(torch.__version__) >= version.parse("1.4"):
                        lr = self.lr_scheduler.get_last_lr()[0]
                    else:
                        lr = self.lr_scheduler.get_lr()[0]
                else:
                    lr=self.optim.param_groups[-1]['lr']

                for k, v in results.items():  
                    if k in epoch_results:
                        if isinstance(v, int):
                            epoch_results[k] += v
                        elif isinstance(v, torch.Tensor):
                            epoch_results[k] += v.item()

                if self.verbose and step_i % 1==0:    
                    desc_str = f'Epoch {epoch} | LR {lr:.6f} |'   

                    for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):

                        if loss_name in results: 
                            loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                        if len(loss_meter) > 0:
                            loss_count = epoch_results[f'{loss_name}_count']    
                            desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'

                    pbar.set_description(desc_str)
                    pbar.update(1)
                dist.barrier()

            if self.verbose:
                pbar.close()

            dist.barrier()

            results = reduce_dict(epoch_results,average=False)    # For global information

            dist.barrier()


            if self.verbose:
                train_loss = results['total_loss']
                train_loss_count = results['total_loss_count']

                avg_train_loss = train_loss / train_loss_count
                losses_str = f"Train Loss: {avg_train_loss:.3f}\n"

                for name, loss in results.items():
                    if name[-4:] == 'loss':
                        loss_count = int(results[name+'_count'])
                        if loss_count > 0:
                            avg_loss = loss/loss_count
                            losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "

                losses_str += '\n'
                print(losses_str)      #print once per epoch

            dist.barrier()

            if self.verbose:   
                for ig in range(32):
                    save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_A.default.weight.data,"./llama_{}_end/Gqa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                    save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_A.default.weight.data,"./llama_{}_end/Gka_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                    save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_A.default.weight.data,"./llama_{}_end/Gva_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                    save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_A.default.weight.data,"./llama_{}_end/Goa_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                    save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.q_proj.lora_B.default.weight.data,"./llama_{}_end/Gqb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                    save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.k_proj.lora_B.default.weight.data,"./llama_{}_end/Gkb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                    save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.v_proj.lora_B.default.weight.data,"./llama_{}_end/Gvb_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                    save_pickle(self.model.module.base_model.model.model.layers[ig].self_attn.o_proj.lora_B.default.weight.data,"./llama_{}_end/Gob_{}_{}.pkl".format(epoch+1,ig,self.args.lr))
                save_pickle(self.model.module.base_model.model.lm_head.lora_A.default.weight,"./llama_{}_end/Glm_a_{}.pkl".format(epoch+1,self.args.lr))
                save_pickle(self.model.module.base_model.model.lm_head.lora_B.default.weight,"./llama_{}_end/Glm_b_{}.pkl".format(epoch+1,self.args.lr))

                torch.save(self.first_model.state_dict(),"Gfirst_{}_{}_8_{}_end.pth".format(epoch+1,self.args.lr,self.args.train))
            dist.barrier()
            

    def test(self):   
        for epoch in range(8*self.args.epoch):
            torch.cuda.empty_cache()
            
            if (epoch+1)%8==1:
                doc_prefix='./llama_{}_mmid1/'.format(epoch//8+1)
                ckpt_path = "Gfirst_{}_{}_8_{}_mmid1.pth".format(epoch//8+1,self.args.lr,self.args.train) 
            elif (epoch+1)%8==2:
                doc_prefix='./llama_{}_mid1/'.format(epoch//8+1)
                ckpt_path = "Gfirst_{}_{}_8_{}_mid1.pth".format(epoch//8+1,self.args.lr,self.args.train) 
            elif (epoch+1)%8==3:
                doc_prefix='./llama_{}_mmid2/'.format(epoch//8+1)
                ckpt_path = "Gfirst_{}_{}_8_{}_mmid2.pth".format(epoch//8+1,self.args.lr,self.args.train) 
            elif (epoch+1)%8==4:
                doc_prefix='./llama_{}_mid2/'.format(epoch//8+1)
                ckpt_path = "Gfirst_{}_{}_8_{}_mid2.pth".format(epoch//8+1,self.args.lr,self.args.train)
            elif (epoch+1)%8==5:
                doc_prefix='./llama_{}_mmid3/'.format(epoch//8+1)
                ckpt_path = "Gfirst_{}_{}_8_{}_mmid3.pth".format(epoch//8+1,self.args.lr,self.args.train)
            elif (epoch+1)%8==6:
                doc_prefix='./llama_{}_mid3/'.format(epoch//8+1)
                ckpt_path = "Gfirst_{}_{}_8_{}_mid3.pth".format(epoch//8+1,self.args.lr,self.args.train)
            elif (epoch+1)%8==7:
                doc_prefix='./llama_{}_mend/'.format(epoch//8+1)
                ckpt_path = "Gfirst_{}_{}_8_{}_mend.pth".format(epoch//8+1,self.args.lr,self.args.train)
            else:
                doc_prefix='./llama_{}_end/'.format(epoch//8+1)
                ckpt_path = "Gfirst_{}_{}_8_{}_end.pth".format(epoch//8+1,self.args.lr,self.args.train)

            #One can directly assign the checkpoint here when testing.
            ##doc_prefix='./your_main_model_pickle_fold/'
            ##ckpt_path='your_first_model.pth'

            self.load_checkpoint(ckpt_path)
            self.first_model=self.first_model.to(self.args.gpu)

            for gg in range(32):
                self.model.base_model.model.model.layers[gg].self_attn.q_proj.lora_A.default.weight.data=load_pickle(doc_prefix+"Gqa_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                self.model.base_model.model.model.layers[gg].self_attn.k_proj.lora_A.default.weight.data=load_pickle(doc_prefix+"Gka_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                self.model.base_model.model.model.layers[gg].self_attn.v_proj.lora_A.default.weight.data=load_pickle(doc_prefix+"Gva_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                self.model.base_model.model.model.layers[gg].self_attn.o_proj.lora_A.default.weight.data=load_pickle(doc_prefix+"Goa_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                self.model.base_model.model.model.layers[gg].self_attn.q_proj.lora_B.default.weight.data=load_pickle(doc_prefix+"Gqb_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                self.model.base_model.model.model.layers[gg].self_attn.k_proj.lora_B.default.weight.data=load_pickle(doc_prefix+"Gkb_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                self.model.base_model.model.model.layers[gg].self_attn.v_proj.lora_B.default.weight.data=load_pickle(doc_prefix+"Gvb_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
                self.model.base_model.model.model.layers[gg].self_attn.o_proj.lora_B.default.weight.data=load_pickle(doc_prefix+"Gob_{}_{}.pkl".format(gg,self.args.lr)).to(args.gpu)
            self.model.base_model.model.lm_head.lora_A.default.weight.data=load_pickle(doc_prefix+"Glm_a_{}.pkl".format(self.args.lr)).data.to(args.gpu)
            self.model.base_model.model.lm_head.lora_B.default.weight.data=load_pickle(doc_prefix+"Glm_b_{}.pkl".format(self.args.lr)).data.to(args.gpu)
            if self.verbose:
                print('Main model loaded.')
            self.model = self.model.to(self.args.gpu)
            
            valid_results = self.evaluate_epoch()   # For accuracy
            dist.barrier()

            valid_results = new_reduce_dict(valid_results)   
            dist.barrier()

            if self.verbose:
                print()
                print()
                for kk in valid_results.keys():
                    if kk.endswith('transductive'):
                        if self.args.train=='Arxiv':
                            valid_results[kk]=valid_results[kk].item()/self.val_loader.dataset.len_transductive
                print(valid_results)
                print()
                print()

            dist.barrier()

            if self.verbose:
                acc_file=open('Llama_7b.txt','a')                 
                if (epoch+1)%8==1:
                    acc_file.write(str(epoch//8+1)+'_mmid1'+'\n')
                elif (epoch+1)%8==2:
                    acc_file.write(str(epoch//8+1)+'_mid1'+'\n')
                elif (epoch+1)%8==3:
                    acc_file.write(str(epoch//8+1)+'_mmid2'+'\n')
                elif (epoch+1)%8==4:
                    acc_file.write(str(epoch//8+1)+'_mid2'+'\n')
                elif (epoch+1)%8==5:
                    acc_file.write(str(epoch//8+1)+'_mmid3'+'\n')
                elif (epoch+1)%8==6:
                    acc_file.write(str(epoch//8+1)+'_mid3'+'\n')
                elif (epoch+1)%8==7:
                    acc_file.write(str(epoch//8+1)+'_mend'+'\n')
                else:
                    acc_file.write(str(epoch//8+1)+'_end'+'\n')
                acc_file.write(str(valid_results)+'\n\n')
                acc_file.close()
            dist.barrier()


    def evaluate_epoch(self):   
        ACC={}
        for k in list(self.val_list.keys()):
            if k=='link':
                pass
            elif k=='classification':
                if self.args.train=='Arxiv':
                    templates=[]
                    for tems in self.val_list[k]:
                        templates=templates+tems
                    for thing in templates:
                        ACC[thing+'-'+'transductive']=0
        self.model.eval()
        self.first_model.eval()
        with torch.no_grad():
            for step_i, batch in tqdm(enumerate(self.val_loader)):  
                torch.cuda.empty_cache()

                if self.args.distributed:
                    device = next(self.model.parameters()).device
                    input_ids = batch['input_ids'].to(device)
                    embeds = self.first_model(input_ids=input_ids)
                    attention_mask=batch['attn_mask'].to(device)
                    results = self.model.g_step(in_embeds=embeds, attention_mask=attention_mask)

                for iiid in range(len(results)):    
                    task=batch['task'][iiid]
                    temp_id=batch['temp_ids'][iiid]
                    if task=='classification':
                        cate=batch['cate'][iiid] 
                        if temp_id.endswith('2') or temp_id.endswith('4') or temp_id.endswith('6') or temp_id.endswith('7'): 
                            if results[iiid].lower()==batch['target_text'][iiid]:
                                ACC[temp_id+'-'+cate]+=1
                                #Check if the generated text strings is strictly matched with the label in natural language.
                        else:
                            pass

                    elif task=='link':
                        pass

                dist.barrier()

            return ACC   


def main_worker(gpu, args):     # the gpu is the local_rank
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    # define the prompts used in training
    if not args.inference:                      # Train Consoles
        print(f'Building train loader at GPU {gpu}')
        # The '6-6-6-7' represents the graph-free instruction prompt. 
        # All detailed instruction prompt lists are summarized in all_graph_templates.py and the appendix of our paper.
        if args.train=='Arxiv':
            train_task_list = {
            'classification' : [['6-6-6-7'],['2-1-3-2','2-1-3-4']],
            # 'classification':[['6-6-6-7'],['2-1-1-2','2-3-1-2'],['2-1-2-2','2-1-2-4','2-3-2-2','2-3-2-4'],['2-1-3-2','2-1-3-4','2-3-3-2','2-3-3-4']],
            'link':[['1-1-3-1']]
            # 'link':[['1-1-1-1','1-3-1-1'],['1-1-2-1','1-1-2-3','1-3-2-1','1-3-2-3'],['1-1-3-1','1-1-3-3','1-3-3-1','1-3-3-3']]
            }

        train_sample_numbers = {} # Abandoned

        train_loader = get_loader(
            args,
            train_task_list,
            train_sample_numbers,
            split=args.train, 
            mode='train',
            batch_size=args.batch_size,
            workers=args.num_workers,
            distributed=args.distributed
        )

        if args.gpu==0:
            print('Length of train dataset:', len(train_loader.dataset))
        trainer = Trainer(args,train_loader= train_loader,  train=True)   
        trainer.train()

    # define the prompts used in validation/ Test
    if args.inference:                              # Valid/ Test Consoles
        print(f'Building val loader at GPU {gpu}')
        # The '6-6-6-7' represents the graph-free instruction prompt. 
        if args.valid=='Arxiv':
            val_task_list = {
            'classification':[['6-6-6-7']]#,['2-1-1-2','2-3-1-2'],['2-1-2-2','2-1-2-4','2-3-2-2','2-3-2-4'],['2-1-3-2','2-1-3-4','2-3-3-2','2-3-3-4']]
            }
        val_sample_numbers = {}  # Abandoned
        val_loader = get_loader(
            args,
            val_task_list,
            val_sample_numbers,
            split=args.valid, 
            mode='val',
            batch_size=args.batch_size,
            workers=args.num_workers,
            distributed=args.distributed
        )

        if args.gpu==0:
            print('Length of test dataset:', len(val_loader.dataset))

        trainer = Trainer(args, val_loader= val_loader, train=False,val_list=val_task_list)   
        trainer.test()


if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank in [0, -1]:
        print(args)

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node

    LOSSES_NAME = [f'{name}_loss' for name in args.losses.split(',')]
    if local_rank in [0, -1]:
        print(LOSSES_NAME)
    LOSSES_NAME.append('total_loss')

    args.LOSSES_NAME = LOSSES_NAME

    comments = []
    dsets = []
    if 'Arxiv' in args.train:
        dsets.append('Arxiv')
    comments.append(''.join(dsets))

    comments.append(args.backbone)
    comments.append(''.join(args.losses.split(',')))
    if args.comment != '':
        comments.append(args.comment)
    comment = '_'.join(comments)

    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M')

    project_dir = Path(__file__).resolve().parent.parent

    if local_rank in [0, -1]:
        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'
        args.run_name = run_name

    if args.distributed:
        main_worker(local_rank, args)
