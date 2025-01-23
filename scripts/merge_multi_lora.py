# Usage: python merge_checkpoints.py chekpoint_path1 checkpoint_path2 -o merged_checkpoint_path

from collections import defaultdict
import os
import json
import torch
import argparse
from tqdm import tqdm
from safetensors.torch import load_file, save_file
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

# from ties_merging import do_merging, convert_delta_to_ft
# from calculate_metrics import calculate_metrics

MODAL_DICT={'mm_cf_tower': 'cf',
            'mm_vision_tower': 'vision'}
def get_modal_from_config(config):
    for key in MODAL_DICT:
        if key in config.keys() and isinstance(config[key], str) and len(config[key]) > 0:
            return MODAL_DICT[key]
    assert False, f'No modality is recognized, please check the config.'

def merge_checkpoints(filepaths, output_path, args, strategy="sum", K=20):
    configs = []
    adapter_configs = []
    weights_to_merge = defaultdict(list)
    proj_to_merge = defaultdict(list) if args.merge_proj else None
    for filepath in filepaths:
        adapter_path = os.path.join(filepath, 'adapter_model.safetensors')
        adapter_weights = load_file(adapter_path, device="cpu")
        modal_config = json.load(open(os.path.join(filepath, 'config.json')))
        adapter_config = json.load(open(os.path.join(filepath, 'adapter_config.json')))
        if args.merge_proj:
            projector_weights = torch.load(os.path.join(filepath, 'non_lora_trainables.bin'))
        configs.append(modal_config)
        adapter_configs.append(adapter_config)
        
        for key in adapter_weights:
            weights_to_merge[key].append(adapter_weights[key])
        if args.merge_proj:
            for key in projector_weights:
                proj_to_merge[key].append(projector_weights[key])
    
    if strategy.startswith('online-merge-'):
        merged_weights = dict()
        modal_names = [get_modal_from_config(config) for config in configs]
        for key in weights_to_merge:
            if len(weights_to_merge[key]) == 1:
                merged_weights[key] = weights_to_merge[key][0]
            else:
                assert 'default' in key
                for modal_name, weight in zip(modal_names, weights_to_merge[key]):
                    merged_weights[key.replace('default', f'default-{modal_name}')] = weight
    else:
        if strategy == 'sum':
            merged_weights = {}
            for key in weights_to_merge:
                merged_weights[key] = sum(weights_to_merge[key])
        elif strategy == 'mean':
            merged_weights = {}
            for key in weights_to_merge:
                merged_weights[key] = sum(weights_to_merge[key]) / len(weights_to_merge[key])
            if args.merge_proj:
                merged_proj = {}
                for key in proj_to_merge:
                    merged_proj[key] = sum(proj_to_merge[key]) / len(proj_to_merge[key])
        elif strategy == 'weighted_sum':
            merged_weights = {}
            if args.weights is None:
                raise ValueError("Weights not provided for weighted sum")
            weights = args.weights.split(',')
            weights = [float(weight) for weight in weights]
            for key in weights_to_merge:
                merged_weights[key] = sum([weight * w for weight, w in zip(weights, weights_to_merge[key])])
            if args.merge_proj:
                merged_proj = {}
                for key in proj_to_merge:
                    merged_proj[key] = sum([weight * w for weight, w in zip(weights, proj_to_merge[key])])
        else:
            print(f"Merge strategy [{strategy}] not implemented, DO NOTHING.")
            # raise NotImplementedError("Merge strategy not implemented")
    merged_configs = {}
    for config in configs:
        for key in config:
            if key in merged_configs:
                merged_configs[key] = merged_configs[key] or config[key]
            else:
                merged_configs[key] = config[key]
        # if strategy.startswith('online-merge-'):
        #     strategy = strategy.replace('online-merge-', '')
        #     if strategy.startswith('reset-'):
        #         merged_configs['reset_scaling_weights'] = strategy.replace('reset-', '')
        #     else:
        #         merged_configs['merge_default_weights'] = strategy
    
    merged_adapter_configs = {}
    for adapter_config in adapter_configs:
        for key in adapter_config:
            if key in merged_adapter_configs:
                merged_adapter_configs[key] = merged_adapter_configs[key] or adapter_config[key]
            else:
                merged_adapter_configs[key] = adapter_config[key]
    
    # for config in configs:
    #     modal_name = get_modal_from_config(config)
    #     # lora_r_dict[modal_name] = config.r
    #     # lora_alpha_dict[modal_name] = config.lora_alpha
    #     merged_configs[f'{modal_name}_lora_alpha'] = config['lora_alpha']
    #     merged_configs[f'{modal_name}_lora_r'] = config['lora_r']
    
    os.makedirs(output_path, exist_ok=True)
    save_file(merged_weights, os.path.join(output_path, 'adapter_model.safetensors'))
    json.dump(merged_configs, open(os.path.join(output_path, 'config.json'), 'w'), indent=4)
    json.dump(merged_adapter_configs, open(os.path.join(output_path, 'adapter_config.json'), 'w'), indent=4)
    if args.merge_proj:
        torch.save(merged_proj, os.path.join(output_path, 'non_lora_trainables.bin'))
    else:
        # save mm_projector.bin
        for filepath in filepaths:
            projector_path = os.path.join(filepath, 'non_lora_trainables.bin')
            if os.path.exists(projector_path):
                projector_weights = torch.load(projector_path, map_location='cpu')
                if "cf" in filepath:
                    torch.save(projector_weights, os.path.join(output_path, 'non_lora_trainables_cf.bin'))
                elif "image" in filepath:
                    torch.save(projector_weights, os.path.join(output_path, 'non_lora_trainables.bin'))
            else:
                continue
        
    
    # with open(os.path.join(output_path, 'merge_info.txt'), 'w') as fout:
    #     inputs = '\n'.join(filepaths)
    #     fout.write(f"Inputs:\n{inputs}\n\nOutput({strategy}):{output_path}")
    # print(f"Merged checkpoints saved to {output_path}")
    
    # # calculate merge metrics
    # if ft_checks:
    #     calculate_metrics(output_path, reset_thresh=K)

def main():
    parser = argparse.ArgumentParser(description='Merge multiple torch checkpoints')
    parser.add_argument('filepaths', nargs='+', help='List of checkpoint file paths to merge')
    parser.add_argument('-o', '--output', default='merged_checkpoint.pth', help='Output file path')
    parser.add_argument('--strategy', default='sum', help='Merge strategy')
    parser.add_argument('-K', default=20, type=int, help='K for ties-merging')
    parser.add_argument('--weights', default=None, help='Weight to merge')
    parser.add_argument('--merge_proj', action='store_true', help='Merge projector')
    args = parser.parse_args()

    merge_checkpoints(args.filepaths, args.output, args, args.strategy, args.K)

if __name__ == '__main__':
    main()