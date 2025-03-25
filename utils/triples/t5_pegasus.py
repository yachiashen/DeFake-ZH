import jieba
from transformers import BertTokenizer
from collections.abc import Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
import re
import os
import argparse
from tqdm.auto import tqdm
import numpy as np
import rouge
from opencc import OpenCC

from ..const import PROJECT_FOLDER

# class T5PegasusTokenizer(BertTokenizer):
#     """结合中文特点完善的Tokenizer
#     基于词颗粒度的分词，如词表中未出现，再调用BERT原生Tokenizer
#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     def pre_tokenizer(self, x):
#         return jieba.cut(x, HMM=False)
#
#     def _tokenize(self, text, *arg, **kwargs):
#         split_tokens = []
#         for text in self.pre_tokenizer(text):
#             if text in self.vocab:
#                 split_tokens.append(text)
#             else:
#                 split_tokens.extend(super()._tokenize(text))
#         return split_tokens
#
#
# class KeyDataset(Dataset):
#     def __init__(self, dict_data):
#         self.data = dict_data
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index):
#         return self.data[index]
#
#
# def create_data(data, tokenizer, max_len):
#     """调用tokenizer.encode编码正文/标题，每条样本用dict表示数据域
#     """
#     ret, flag, title = [], True, None
#     for content in data:
#         if type(content) == tuple:
#             title, content = content
#         text_ids = tokenizer.encode(content, max_length=max_len,
#                                     truncation='only_first')
#
#         if flag:
#             flag = False
#
#         features = {'input_ids': text_ids,
#                     'attention_mask': [1] * len(text_ids),
#                     'raw_data': content}
#         if title:
#             features['title'] = title
#         ret.append(features)
#     return ret
#
#
# def sequence_padding(inputs, length=None, padding=0):
#     """Numpy函数，将序列padding到同一长度
#     """
#     if length is None:
#         length = max([len(x) for x in inputs])
#
#     pad_width = [(0, 0) for _ in np.shape(inputs[0])]
#     outputs = []
#     for x in inputs:
#         x = x[:length]
#         pad_width[0] = (0, length - len(x))
#         x = np.pad(x, pad_width, 'constant', constant_values=padding)
#         outputs.append(x)
#
#     return np.array(outputs, dtype='int64')
#
#
# def default_collate(batch):
#     """组batch
#     各个数据域分别转换为tensor，tensor第一个维度等于batch_size
#     """
#     np_str_obj_array_pattern = re.compile(r'[SaUO]')
#     default_collate_err_msg_format = (
#         "default_collate: batch must contain tensors, numpy arrays, numbers, "
#         "dicts or lists; found {}")
#     elem = batch[0]
#     elem_type = type(elem)
#     if isinstance(elem, torch.Tensor):
#         out = None
#         if torch.utils.data.get_worker_info() is not None:
#             # If we're in a background process, concatenate directly into a
#             # shared memory tensor to avoid an extra copy
#             numel = sum([x.numel() for x in batch])
#             storage = elem.storage()._new_shared(numel)
#             out = elem.new(storage)
#         return torch.stack(batch, 0, out=out).to(device)
#     elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
#             and elem_type.__name__ != 'string_':
#         if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
#             # array of string classes and object
#             if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
#                 raise TypeError(default_collate_err_msg_format.format(elem.dtype))
#
#             return default_collate([torch.as_tensor(b) for b in batch])
#         elif elem.shape == ():  # scalars
#             return torch.as_tensor(batch)
#     elif isinstance(elem, float):
#         return torch.tensor(batch, dtype=torch.float64)
#     elif isinstance(elem, int):
#         return torch.tensor(batch, dtype=torch.long)
#     elif isinstance(elem, str):
#         return batch
#     elif isinstance(elem, Mapping):
#         return {key: default_collate([d[key] for d in batch]) for key in elem}
#     elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
#         return elem_type(*(default_collate(samples) for samples in zip(*batch)))
#     elif isinstance(elem, Sequence):
#         # check to make sure that the elements in batch have consistent size
#         it = iter(batch)
#         elem_size = len(next(it))
#         if not all(len(elem) == elem_size for elem in it):
#             batch = sequence_padding(batch)
#
#         return default_collate([default_collate(elem) for elem in batch])
#
#     raise TypeError(default_collate_err_msg_format.format(elem_type))
#
#
# def prepare_data(args, tokenizer, input_data):
#     """准备batch数据
#     """
#     input_data = create_data(input_data, tokenizer, args.max_len)
#     input_data = KeyDataset(input_data)
#     input_data = DataLoader(input_data, batch_size=args.batch_size, collate_fn = default_collate)
#     return input_data
#
#
# def generate(input_data, model, tokenizer, args):
#     gens = []
#
#     model.eval()
#     for feature in tqdm(input_data):
#         raw = feature['raw_data']
#         content = {k : v for k, v in feature.items() if k not in ['raw_data', 'title']}
#         gen = model.generate(max_length = args.max_len_generate,
#                             eos_token_id = tokenizer.sep_token_id,
#                             decoder_start_token_id=tokenizer.cls_token_id,
#                             **content)
#         gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
#         gen = [item.replace(' ', '') for item in gen]
#         gens.extend(gen)
#
#     return gens
#
# def init_argument():
#     parser = argparse.ArgumentParser(description='t5-pegasus-chinese')
#     parser.add_argument('--pretrain_model', default=os.path.join(PROJECT_FOLDER, 't5_pegasus_pretrain'))
#     parser.add_argument('--model', default=os.path.join(PROJECT_FOLDER, 'saved_model', 'summary_model'))
#
#     parser.add_argument('--batch_size', default = 16, help='batch size')
#     parser.add_argument('--max_len', default = 512, help='max length of inputs')
#     parser.add_argument('--max_len_generate', default = 100, help='max length of generated text')
#     parser.add_argument('--use_multiprocess', default = False, action = 'store_true')
#
#     args = parser.parse_args()
#     return args
#
# def language_transform(input_data, config):
#     cc = OpenCC(config)
#     ret_data = []
#     for i in range(len(input_data)):
#         ret_data.append(cc.convert(input_data[i]))
#
#     return ret_data
#
# def coreference(input_data: list[str]):
#
#     # input_data = language_transform(input_data, 't2s')
#     input_data = prepare_data(args, tokenizer, input_data)
#     gens = generate(input_data, t5_pegasus_model, tokenizer, args)
#
#     # gens = language_transform(gens, 's2t')
#
#     return gens
#
#
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# args = init_argument()
# tokenizer = T5PegasusTokenizer.from_pretrained(args.pretrain_model)
# t5_pegasus_model = torch.load(args.model, map_location = device, weights_only = False)
#
#
# if __name__ == '__main__':
#     pass