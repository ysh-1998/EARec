import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from mllm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, ITEM_TOKEN_INDEX
from mllm.conversation import conv_templates, SeparatorStyle
from mllm.model.builder import load_pretrained_model
from mllm.utils import disable_torch_init
from mllm.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, tokenizer_item_token, tokenizer_modal_token

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def gather_indexes(output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,data_path=args.question_file)
    questions = json.load(open(args.question_file, "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    embedding_list = []
    for line in tqdm(questions):
        item = [int(line["item"])] if "item" in line else []
        image_file = [line["image"]] if "image" in line else []

        question = line["conversations"][0]["value"]
        # gt = line["conversations"][1]["value"]

        if len(image_file) == 0 and len(item) == 0:
            inputs = tokenizer([question])
            input_ids = torch.as_tensor(inputs.input_ids).cuda()
            with torch.inference_mode():
                outputs = model(
                    input_ids,
                    output_hidden_states=True
                )
        else:
            image_tensor = None
            item_tensor = None
            input_ids = tokenizer_modal_token(question, tokenizer, return_tensors='pt').unsqueeze(0).cuda()
            if len(image_file) == 1:
                image = Image.open(os.path.join(args.image_folder, image_file[0])).convert('RGB')
                image_tensor = process_images([image], image_processor, model.config).half().cuda()
            if len(image_file) > 1:
                image = [Image.open(os.path.join(args.image_folder, x)).convert('RGB') for x in image_file]
                image_tensor = [process_images([x], image_processor, model.config)[0].half().cuda() for x in image]
                image_tensor = torch.stack(image_tensor, dim=0)
            if len(item) == 1:
                item_tensor = torch.tensor(item,dtype=torch.long).unsqueeze(0).cuda()
            if len(item) > 1:
                item_tensor = torch.tensor(item,dtype=torch.long).cuda()
            with torch.inference_mode():
                outputs = model(
                    input_ids,
                    images=image_tensor,
                    # image_sizes=[image[0].size] if image_tensor is not None else None,
                    items=item_tensor,
                    output_hidden_states=True
                )
        # save last hidden state as embeddings
        if args.pooling == "first-last-avg":
            last_hidden_states = outputs.hidden_states[-1].squeeze(0).detach().cpu()#float()
            first_hidden_states = outputs.hidden_states[0].squeeze(0).detach().cpu()#.float()
            first_last_avg_states = torch.mean(first_hidden_states+last_hidden_states, dim=0).unsqueeze(0)
            embedding_list.append(first_last_avg_states)
        if args.pooling == "last":
            embedding = outputs.hidden_states[-1][0,-1,:].unsqueeze(0).detach().cpu()#.float()
            embedding_list.append(embedding)
        if args.pooling == "last-avg":
            embedding = torch.mean(outputs.hidden_states[-1][0,:,:], dim=0).unsqueeze(0).detach().cpu()#.float()
            embedding_list.append(embedding)
    embeddings = torch.cat(embedding_list, dim=0).numpy().astype('float32')
    print('Embeddings shape: ', embeddings.shape)
    file = os.path.join(answers_file)
    embeddings.tofile(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--only_title", type=int, default=0)
    parser.add_argument("--pooling", type=str, default="last")
    parser.add_argument("--image_pooling", type=str, default="patch")
    args = parser.parse_args()

    eval_model(args)
