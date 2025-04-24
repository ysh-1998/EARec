# EARec
The official implementation for *Explainable Multi-Modality Alignment for
Transferable Recommendation*. [paper](blob/The_WebConf_2025_.pdf)

> Shenghao Yang, Weizhi Ma, Zhiqiang Guo, Min Zhang, Haiyang Wu, Junjie Zhai, Chunhui Zhang, Yuekui Yang. Explainable Multi-Modality Alignment for Transferable Recommendation. TheWebConf 2024.


<div align=center>
<img src="Figure/EARec.png" alt="EARec" width="100%" />
</div>

## Quick Start

To repreduce our experiment results of ```Office``` dataset, you should follow the steps below:

1.Git clone this repository.
```
git clone https://github.com/ysh-1998/EARec.git
```
We provide a preprocessed dataset of ```Office``` dataset in ```dataset/Office```. The images of ```Office``` dataet should be download from [Google Drive](https://drive.google.com/file/d/1r2-s-iXU97-MuUdETM-t5m_R88064u5J/view?usp=drive_link) and place in ```dataset/Office/image```. 

2.Install the required packages. The enviorment is built following [LLaVA](https://github.com/haotian-liu/LLaVA), please refer to the steps of [link](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#install).

3.Download the finetuned LoRA weights of three modalities from [Google Drive](https://drive.google.com/file/d/13ybxzRQwVa99TkGfEaS4sqijAz02-EzJ/view?usp=drive_link) and place them in ```checkpoints```. To reproduce the alignment training process, you can refer to the scripts in ```scripts/v1_5```.

4.Merge the LoRA weights of three modalities.
```
bash scripts/v1_5/merge_lora.sh
```
5.Get the multi-modal item embedding with EARec.
```
bash scripts/v1_5/eval/eval_emb.sh
```
You can also get the multi-modal item embedding with mulitple GPUs using following command.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/v1_5/eval/eval_emb_dp.sh
```
6.Train and evaluate downstream recommendation model with aligned multi-modal item embedding.
```
python downstream/finetune.py --gpu_id=0 -d Office --plm_suffix="featEARec.tvb"
```