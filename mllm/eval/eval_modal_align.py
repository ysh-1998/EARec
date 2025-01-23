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
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, data_path=args.question_file)
    questions = json.load(open(args.question_file, "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    if args.only_title:
        answers_file = answers_file.replace('.jsonl', '_only_title.jsonl')
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    scores = []
    count = 0
    for line in tqdm(questions):
        item = [int(line["item"])] if "item" in line else []
        also_buy_ids = eval(line['also_buy']) if 'also_buy' in line else []
        also_view_ids = eval(line['also_view']) if 'also_view' in line else []
        item = item + also_buy_ids + also_view_ids
        image_file = [line["image"]] if "image" in line else []
        also_image = eval(line['also_image']) if 'also_image' in line else []
        image_file = image_file + also_image

        question = line["conversations"][0]["value"]
        gt_answer = line["conversations"][1]["value"]
        if args.only_title:
            gt_answer = gt_answer.split('Category:')[0]
            gt_answer = gt_answer[7:]

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        question = conv.get_prompt()
        
        image_tensor = None
        item_tensor = None
        if len(image_file) == 0 and len(item) == 0:
            inputs = tokenizer([question])
            input_ids = torch.as_tensor(inputs.input_ids).cuda()
        else:
            input_ids = tokenizer_modal_token(question, tokenizer, return_tensors='pt').unsqueeze(0).cuda()
            if len(image_file) == 1:
                image = Image.open(os.path.join(args.image_folder, image_file[0])).convert('RGB')
                image_tensor = process_images([image], image_processor, model.config).half().cuda().unsqueeze(0)
            if len(image_file) > 1:
                image = [Image.open(os.path.join(args.image_folder, x)).convert('RGB') for x in image_file]
                image_tensor = [process_images([x], image_processor, model.config)[0].half().cuda() for x in image]
                image_tensor = torch.stack(image_tensor, dim=0).unsqueeze(0)
            if len(item) == 1:
                item_tensor = torch.tensor(item,dtype=torch.long).unsqueeze(0).cuda()
            if len(item) > 1:
                item_tensor = torch.tensor(item,dtype=torch.long).unsqueeze(0).cuda()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                # image_sizes=[image.size] if image_tensor is not None else None,
                items=item_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=64,
                use_cache=True)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if args.only_title:
            outputs = outputs.split('Category:')[0]
            outputs = outputs[7:]
        score = scorer.score(outputs, gt_answer)["rougeL"].fmeasure
        scores.append(score)

        ans_file.write(json.dumps({
                                "id": id,
                                "text": outputs,
                                "gt_text": gt_answer,
                                "score": score}) + "\n")
        ans_file.flush()
        count += 1
        if count == 50:
            break
    avg_rouge = sum(scores) / len(scores)
    ans_file.write(json.dumps({"sum_rouge": sum(scores), "num_rouge": len(scores), "avg_rouge": avg_rouge}) + "\n")
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--only_title", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)
