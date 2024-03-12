import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm, trange
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
# from torch.cuda.amp import autocast
import io
import os
import PIL
import random
import numpy as np
import torch
import torchvision
import transformers
import more_itertools
import numpy as np
import matplotlib.pyplot as plt
# from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset
# from tqdm import tqdm
from dataclasses import dataclass, field
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import cv2
from PIL import Image
import clip


class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    # @autocast()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ClipCaptionModel(nn.Module):
    def __init__(self, gpt, prefix_length: int, prefix_size: int = 768):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        """
        ru gpts shit

        """
        self.gpt = GPT2LMHeadModel.from_pretrained(gpt)
        # self.gpt = freeze(self.gpt)
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                 self.gpt_embedding_size * prefix_length))

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    # @autocast()
    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)

        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


None
import transformers


def filter_ngrams(output_text):
    a_pos = output_text.find(' A:')
    sec_a_pos = output_text.find(' A:', a_pos + 1)
    return output_text[:sec_a_pos]


def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt='',
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.98,
        temperature=1.,
        stop_token='.',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if not tokens:
                tokens = torch.tensor(tokenizer.encode(prompt))
                # print('tokens',tokens)
                tokens = tokens.unsqueeze(0).to(device)

            emb_tokens = model.gpt.transformer.wte(tokens)

            if embed is not None:
                generated = torch.cat((embed, emb_tokens), dim=1)
            else:
                generated = emb_tokens

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                #
                top_k = 2000
                top_p = 0.98
                # print(logits)
                # next_token = transformers.top_k_top_p_filtering(logits.to(torch.int64).unsqueeze(0), top_k=top_k, top_p=top_p)
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)

                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)

                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())

            output_text = tokenizer.decode(output_list)
            output_text = filter_ngrams(output_text)
            generated_list.append(output_text)

    return generated_list[0]


def image_grid(imgs, rows, cols):
    pils = imgs

    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def read_video(path, transform=None, frames_num=9, window=30):
    frames = []
    cap = cv2.VideoCapture(path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    N = length // (frames_num)
    # print(length)
    # counter =

    current_frame = 1
    for i in range(length):

        # frameId = int(round(cap.get(current_frame)))
        # print(current_frame)
        ret, frame = cap.read(current_frame)

        if ret and i == current_frame and len(frames) < frames_num:
            size = 193, 193
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame.thumbnail(size, Image.ANTIALIAS)

            frames.append(frame)
            current_frame += N

        # print(current_frame)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    cap.release()
    # print(frames)
    return frames


# from tqdm import tqdm, trange


def get_caption(model, tokenizer, prefix, prefix_length, prompt=''):
    prefix = prefix.to(device)
    with torch.no_grad():

        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
        if prompt:
            generated_text_prefix = generate2(model, tokenizer, prompt=prompt, embed=prefix_embed)
        else:
            generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)
    return generated_text_prefix.replace('\n', ' ')


def get_ans(model, tokenizer, clip_emb, prefix_length, prompt):
    output = get_caption(model, tokenizer, clip_emb, prefix_length, prompt=prompt)
    ans = output[len(prompt):].strip()
    return {'answer': ans}


import json
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import single_meteor_score
import pandas as pd
import json
import random
import pandas as pd
import re
import nltk.translate.bleu_score as bleu


def lus(string):
    # Creating a set to store the last positions of occurrence
    seen = {}
    maximum_length = 0
    max_end = 0
    max_start = 0

    # starting the initial point of window to index 0
    start = 0

    for end in range(len(string)):

        # Checking if we have already seen the element or not
        if string[end] in seen:
            # If we have seen the number, move the start pointer
            # to position after the last occurrence
            start = max(start, seen[string[end]] + 1)

        # Updating the last seen value of the character
        seen[string[end]] = end
        if end - start + 1 > maximum_length:
            maximum_length = end - start + 1
            max_end = end + 1
            max_start = start
        # maximum_length = max(maximum_length, end-start + 2)

    return maximum_length, max_start, max_end


def clean_str(string):
    maxlen, start, end = lus(string)
    # print(maxlen, start, end)
    substr = string[start:end]
    # print(substr)
    first_pos = string.find(substr)
    sec_pos = string.find(substr, first_pos + 1)
    ans = string[:sec_pos].strip()
    if sec_pos == -1:
        ans = string[:first_pos + len(substr)].strip()
    point = ans.find('.')
    if punto != -1:
        ans = ans[:point]
    return ans


def lrs(str):
    n = len(str)
    LCSRe = [[0 for x in range(n + 1)]
             for y in range(n + 1)]

    res = ""  # To store result
    res_length = 0  # To store length of result

    # building table in bottom-up manner
    index = 0
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):

            # (j-i) > LCSRe[i-1][j-1] to remove
            # overlapping
            if (str[i - 1] == str[j - 1] and
                    LCSRe[i - 1][j - 1] < (j - i)):
                LCSRe[i][j] = LCSRe[i - 1][j - 1] + 1

                # updating maximum length of the
                # substring and updating the finishing
                # index of the suffix
                if (LCSRe[i][j] > res_length):
                    res_length = LCSRe[i][j]
                    index = max(i, index)

            else:
                LCSRe[i][j] = 0

    # If we have non-empty result, then insert
    # all characters from first character to
    # last character of string
    if (res_length > 0):
        for i in range(index - res_length + 1,
                       index + 1):
            res = res + str[i - 1]

    return res


def is_b2b(full, sub):
    first_oc = full.find(sub)
    sec_oc = full.find(sub, first_oc + 1)
    if sub == full or sec_oc == (first_oc + len(sub)) or sec_oc == (first_oc + len(sub) + 1):
        return True
    return False


def find_sub(string):
    # string = re.sub(' ', '', re_string)
    new_res = string
    results = [new_res]
    while new_res != '':
        new_res = lrs(new_res)
        results.append(new_res)
    i = len(results) - 1
    ans = results[i]
    while not is_b2b(string, ans) or not len(ans) > 1:
        i -= 1
        ans = results[i]
    return ans


def clean_str_lrs(string):
    if len(string) <= 1:
        return string
    prev_res = find_sub(string)

    pos_sub = string.find(prev_res)
    right_pos_sub = string.rfind(prev_res)

    cand_ans_one = string[:pos_sub + len(prev_res)]
    cand_ans_two = string[right_pos_sub:]
    if len(set(cand_ans_one)) >= len(set(cand_ans_two)):
        ans = cand_ans_one
    else:
        ans = cand_ans_two

    punto_sub = ans.find('.')
    if punto_sub != -1:
        ans = ans[:punto_sub]
    return ans.strip()


def eval(preds, results):
    corrects = {i: 0 for i in range(0, 9)}
    meteors = {i: 0 for i in range(0, 9)}

    total_res = preds.merge(results, on='question_id')

    type_count = {}
    for i in range(total_res.shape[0]):
        res = total_res.iloc[i, :]
        res_type = res['type']
        type_count[res_type] = type_count.get(res_type, 0) + 1

    for i in range(total_res.shape[0]):
        pt = total_res.iloc[i, :]
        pt_answer = pt['answer_x']
        pt_question_id = pt['question_id']
        pt_type = pt['type']
        if pt_answer == pt['answer_y']:
            corrects[pt_type] += 1
        tok_pt_ans = word_tokenize(pt_answer)
        tok_pt_result = word_tokenize(pt['answer_y'])
        meteor = single_meteor_score(tok_pt_ans, tok_pt_result)
        meteors[pt_type] += meteor

    return corrects, type_count, meteors


def output(corrects, type_count, meteors):
    all_type_corrects_count = sum(corrects.values())
    free_type_corrects_count = sum(list(corrects.values())[3:])

    all_type_meteors_sum = sum(meteors.values())
    free_type_meteors_sum = sum(list(meteors.values())[3:])

    mean_meteors = {}
    for type_id in meteors:
        mean_meteors[type_id] = meteors[type_id] / float(type_count[type_id])

    accuracy = {}
    for type_id in corrects:
        accuracy[type_id] = corrects[type_id] / float(type_count[type_id])

    all_type_accuracy = all_type_corrects_count / float(sum(type_count.values()))

    all_type_meteor = all_type_meteors_sum / float(sum(type_count.values()))

    free_type_accuracy = free_type_corrects_count / float(sum(list(type_count.values())[3:]))

    free_type_meteor = all_type_meteors_sum / float(sum(list(type_count.values())[3:]))

    all_type_accuracy
    # print ('Accuracy (per question type):')

    # print('\tMotion: {:.04f}\n\tSpatial Relation: {:.04f}\n\tTemporal Relation: {:.04f}\n\tFree: {:.04f}\n\tAll: {:.04f}'.format(accuracy[0], accuracy[1], accuracy[2], free_type_accuracy, all_type_accuracy))
    # print ('Accuracy of the Free type questions(per answer type):')
    # print('\tYes/No: {:.04f}\n\tColor: {:.04f}\n\tObject: {:.04f}\n\tLocation: {:.04f}\n\tNumber: {:.04f}\n\tOther: {:.04f}'.format(accuracy[3], accuracy[4], accuracy[5], accuracy[6], accuracy[7], accuracy[8]))
    # print ('METEOR (per question type):')
    # print('\tMotion: {:.04f}\n\tSpatial Relation: {:.04f}\n\tTemporal Relation: {:.04f}\n\tFree: {:.04f}\n\tAll: {:.04f}'.format(mean_meteors[0], mean_meteors[1], mean_meteors[2], free_type_meteor, all_type_meteor))
    # print ('METEOR of the Free type questions(per answer type):')
    # print('\tYes/No: {:.04f}\n\tColor: {:.04f}\n\tObject: {:.04f}\n\tLocation: {:.04f}\n\tNumber: {:.04f}\n\tOther: {:.04f}'.format(mean_meteors[3], mean_meteors[4], mean_meteors[5], mean_meteors[6], mean_meteors[7], mean_meteors[8]))
    return all_type_accuracy


# gt_file = 'all_type_accuracy'
# pred_file = 'results_en_pref10.json'

# preds = pd.read_json(pred_file)
# preds = preds.dropna()
# results = pd.read_json(gt_file)
# corrects, type_count, meteors = eval(preds, results)
# output(corrects, type_count, meteors)


def main(config):
    prefix_length = config['prefix_len']  # 40
    df_eval = pd.read_csv(config['val'])
    device = 'cuda'
    clip_model, preprocess = clip.load("/app/ViT-L-14-336px.pt", device=device, jit=False)
    clip_model.to(device)

    tokenizer = GPT2Tokenizer.from_pretrained(config['gpt'])

    model_path = config['model']

    model = ClipCaptionPrefix(gpt=config['gpt'], prefix_length=prefix_length)

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)

    out_path = 'Features_val.pkl'

    val_embeddings = []
    val_captions = []
    device = 'cuda'
    for q, p in zip(df_eval.question, df_eval.paths):

        # n= df_eval.iloc[i, 0]#, df_eval.iloc[i, 1]

        text = f'Question:{q}? Answer:'
        path = f'{config["video_path"]}{p}.mp4'
        # print(path)
        try:
            video = read_video(path, transform=None, frames_num=9)
            if len(video) > 0:
                #i = image_grid(video, 2, 2)
                image = torch.stack(list(map(lambda x: preprocess(x).to(device), video)))
                with torch.no_grad():
                    prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
                    prefix = torch.mean(prefix, dim=0).unsqueeze(0)
                    prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
                val_embeddings.append(prefix)
                val_captions.append(text)
        except Exception as e:
            print(e)

    # with open(out_path, 'wb') as f:
    #     pickle.dump({'clip_emb': torch.cat(val_embeddings, dim=0), 'captions': val_captions}, f)
    # from tqdm import tqdm, trange

    answers = []
    for i in tqdm(range(len(val_embeddings))):
        emb = val_embeddings[i]
        caption = val_captions[i]

        # qid = df_eval.iloc[i, 2]
        ans = get_ans(model, tokenizer, emb, prefix_length, caption)
        answers.append(ans['answer'])

    df = pd.DataFrame({'answer': answers})
    df.to_csv(os.path.join(config['output_path'], 'answer.csv'))
    # corrects, type_count, meteors = eval(answers, results)
    # ac = [output(corrects, type_count, meteors)]
    # anss= pd.DataFrame({'acc':ac})
    # anss.to_csv('acc.csv')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', default='private_test.csv', type=str, help='path to test files')
    parser.add_argument('--weight_path', default='./weights/prefix_1-009.pt', type=str, help='checkpoint path')
    parser.add_argument('--output_path', default='./output/', type=str, help='output path')
    parser.add_argument('--video_path', default='./videos_private_test/', type=str, help='path to the videos')
    args = parser.parse_args()

    conf = dict(
        model=args.weight_path,
        video_path=args.video_path,
        val=args.test_path,
        output_path=args.output_path,
        gpt='/app/rugptsmall',
        prefix_len=50
    )

    if torch.cuda.is_available():
        print("Using GPU: {}\n".format(torch.cuda.get_device_name()))
        device = torch.device('cuda')
    else:
        print("\nGPU not found. Using CPU: {}\n".format(platform.processor()))
        device = torch.device('cpu')

    main(conf)
