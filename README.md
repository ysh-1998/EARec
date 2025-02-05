# EARec
The official implementation for *Explainable Multi-Modality Alignment for
Transferable Recommendation*. [paper](blob/The_WebConf_2025_.pdf)

> Shenghao Yang, Weizhi Ma, Zhiqiang Guo, Min Zhang, Haiyang Wu, Junjie Zhai, Chunhui Zhang, Yuekui Yang. Explainable Multi-Modality Alignment for Transferable Recommendation. TheWebConf 2024.


<div align=center>
<img src="Figure/EARec.png" alt="EARec" width="100%" />
</div>

## Requirements

```
python==3.10.14
recbole==1.0.1
torch==2.1.2
cudatoolkit==11.8
transformers==4.37.2
peft==0.11.1
deepspeed==0.12.6
```

## Quick Start

To repreduce our experiment results of ```Office``` dataset, you should follow the steps below:

1.Git clone this repository.
```
git clone https://github.com/ysh-1998/EARec.git
```
We provide a preprocessed dataset of ```Office``` dataset in ```dataset/Office```. The images of ```Office``` dataet should be download from [Google Drive](https://drive.google.com/file/d/1r2-s-iXU97-MuUdETM-t5m_R88064u5J/view?usp=drive_link) and place in ```dataset/Office/image```. 

2.Download the finetuned LoRA weights of three modalities from [Google Drive](https://drive.google.com/file/d/13ybxzRQwVa99TkGfEaS4sqijAz02-EzJ/view?usp=drive_link) and place them in ```checkpoints```. To reproduce the alignment training process, you can refer to the scripts in ```scripts/v1_5```.

3.Merge the LoRA weights of three modalities.
```
bash scripts/v1_5/merge_lora.sh
```
4.Get the multi-modal item embedding with EARec.
```
bash scripts/v1_5/eval/eval_emb.sh
```
You can also get the multi-modal item embedding with mulitple GPUs using following command.
```
bash scripts/v1_5/eval/eval_emb_dp.sh
```
5.Train and evaluate downstream recommendation model with aligned multi-modal item embedding.
```
python downstream/finetune.py --gpu_id=0 -d Office --plm_suffix="featEARec.tvb"
```