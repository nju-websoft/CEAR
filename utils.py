import os
import copy
import torch
import random
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans

def set_random_seed(args):
    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def collate_fn(batch):
    max_len = max([len(sample['input_ids']) for sample in batch])

    input_ids = [sample['input_ids'] + [0]*(max_len-len(sample['input_ids'])) for sample in batch]
    attention_mask = [[1.0]*len(sample['input_ids']) + [0.0]*(max_len-len(sample['input_ids'])) for sample in batch]
    h_index = [sample['h_index'] for sample in batch]
    t_index = [sample['t_index'] for sample in batch]
    labels = [sample['label'] for sample in batch]
    relations = [sample['relation'] for sample in batch]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.float)
    h_index = torch.tensor(h_index, dtype=torch.long)
    t_index = torch.tensor(t_index, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    output = (input_ids, attention_mask, h_index, t_index, labels, relations)
    return output


def compute_cos_sim(tensor_a, tensor_b):
    """
    tensor_a [k, m]
    tensor_b [n, m]
    """
    norm_a = torch.norm(tensor_a, dim=1).unsqueeze(1) # [k, 1]
    norm_b = torch.norm(tensor_b, dim=1).unsqueeze(0) # [1, n]
    cos_sim = torch.mm(tensor_a, tensor_b.T) / torch.mm(norm_a, norm_b) # [k, n]
    return cos_sim


def save_checkpoint(args, model, i_exp, i_task, name):
    if model is None:
        raise Exception(f'The best model of task {i_task} is None.')
    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, args.dataset_name, f"Exp{i_exp}",f"{i_task}_{name}.pkl"))


def get_prototypes(args, model, data, reldata_len):
    model.eval()
    dataloader = DataLoader(data, batch_size=args.test_batch_size, collate_fn=collate_fn, drop_last=False)

    hiddens, features = [], []
    for i, batch in enumerate(dataloader):
        inputs = {
            'input_ids': batch[0].to(args.device),
            'attention_mask': batch[1].to(args.device),
            'h_index': batch[2].to(args.device),
            't_index': batch[3].to(args.device),
        }

        with torch.no_grad():
            hidden, _ = model(**inputs)
            hiddens.append(hidden)

    with torch.no_grad():
        hiddens = torch.cat(hiddens, dim=0)
        hidden_tensors = []
        current_idx = 0
        for i in range(len(reldata_len)):
            rel_len = reldata_len[i]
            rel_hiddens = torch.narrow(hiddens, 0, current_idx, rel_len)
            hidden_proto = torch.mean(rel_hiddens, dim=0)
            hidden_tensors.append(hidden_proto)
            current_idx += rel_len
        hidden_tensors = torch.stack(hidden_tensors, dim=0)
        hidden_tensors = torch.nn.LayerNorm([args.hidden_dim]).to(args.device)(hidden_tensors)
        feature_tensors = model.get_low_dim_feature(hidden_tensors)
    return hidden_tensors, feature_tensors


def memory_select(args, model, data, data_len):
    model.eval()
    dataloader = DataLoader(data, batch_size=args.test_batch_size, collate_fn=collate_fn, drop_last=False, shuffle=False)
    hiddens, memory, memory_len = [], [], []
    for i, batch in enumerate(tqdm(dataloader)):
        inputs = {
            'input_ids': batch[0].to(args.device),
            'attention_mask': batch[1].to(args.device),
            'h_index': batch[2].to(args.device),
            't_index': batch[3].to(args.device),
        }

        with torch.no_grad():
            hidden, _ = model(**inputs)
            hiddens.append(hidden.cpu())
    
    hiddens = np.concatenate(hiddens, axis=0)
    current_len = 0
    for i in range(args.relnum_per_task):
        rel_len = data_len[i]
        kmdata = hiddens[current_len: current_len+rel_len]
        k = min(args.memory_size, rel_len)
        kmeans = KMeans(n_clusters=k, random_state=0)
        distances = kmeans.fit_transform(kmdata)

        rel_data = data[current_len: current_len+rel_len]
        for j in range(k):
            select_idx = np.argmin(distances[:, j]) # [k]
            memory.append(rel_data[select_idx])

        current_len += rel_len
        memory_len.append(k)
    return memory, memory_len


def get_augmentative_data(args, data, data_len):
    index = 0
    data_double = copy.deepcopy(data)
    for i in range(len(data_len)):
        rel_data = data[index: index+data_len[i]]
        index += data_len[i]

        rel_data_temp = copy.deepcopy(rel_data)
        random.shuffle(rel_data_temp)

        for j in range(data_len[i]):
            sample1, sample2 = rel_data[j], rel_data_temp[j]
            input_ids1 = sample1['input_ids'][1:-1]
            input_ids2 = sample2['input_ids'][1:-1]
            h_tokens = input_ids2[input_ids2.index(30522)+1:input_ids2.index(30523)]
            t_tokens = input_ids2[input_ids2.index(30524)+1:input_ids2.index(30525)]
            input_ids1[input_ids1.index(30522)+1: input_ids1.index(30523)] = h_tokens
            input_ids1[input_ids1.index(30524)+1: input_ids1.index(30525)] = t_tokens
            input_ids = [101] + input_ids1 + [102]
            h_index = input_ids.index(30522)
            t_index = input_ids.index(30524)
            data_double.append({
                "input_ids": input_ids,
                'h_index': h_index,
                't_index': t_index,
                'label': sample1['label'],
                'relation': sample1['relation']
            })


    aug_data = []
    add_data1 = copy.deepcopy(data_double)
    random.shuffle(add_data1)
    aug_data1 = rel_data_augment(args, data_double, add_data1)
    aug_data += data_double
    aug_data += aug_data1
    return aug_data


def rel_data_augment(args, rel_data1, rel_data2):
    aug_data = []
    length = min(len(rel_data1), len(rel_data2))
    for i in range(length):
        sample1, sample2 = rel_data1[i], rel_data2[i]
        input_ids1 = sample1['input_ids'][1:-1]
        input_ids2 = sample2['input_ids'][1:-1]
        input_ids2.remove(30522)
        input_ids2.remove(30523)
        input_ids2.remove(30524)
        input_ids2.remove(30525)
        if args.dataset_name == "FewRel":
            length = 512-2-len(input_ids1)
            input_ids2 = input_ids2[:length]
        if i % 2 == 0:
            input_ids = [101] + input_ids1 + input_ids2 + [102]
            h_index = sample1['h_index']
            t_index = sample1['t_index']
        else:
            input_ids = [101] + input_ids2 + input_ids1 + [102]
            h_index = sample1['h_index'] + len(input_ids2)
            t_index = sample1['t_index'] + len(input_ids2)
        aug_data.append({
            "input_ids": input_ids,
            'h_index': h_index,
            't_index': t_index,
            'label': sample1['label'],
            'relation': sample1['relation']
        })
    return aug_data


def get_aca_data(args, data, data_len, current_relations):
    index = 0
    rel_datas = []
    for i in range(len(data_len)):
        rel_data = data[index: index+data_len[i]]
        rel_datas.append(rel_data)
        index += data_len[i]

    rel_id = args.seen_rel_num
    aca_data = copy.deepcopy(data)
    idx = args.relnum_per_task // 2
    for i in range(args.relnum_per_task // 2):
        j = i + idx

        datas1 = rel_datas[i]
        datas2 = rel_datas[j]
        L = 5
        for data1, data2 in zip(datas1, datas2):
            input_ids1 = data1['input_ids'][1:-1]
            e11 = input_ids1.index(30522); e12 = input_ids1.index(30523)
            e21 = input_ids1.index(30524); e22 = input_ids1.index(30525)
            if e21 <= e11:
                continue
            input_ids1_sub = input_ids1[max(0, e11-L): min(e12+L+1, e21)]

            input_ids2 = data2['tokens'][1:-1]
            e11 = input_ids2.index(30522); e12 = input_ids2.index(30523)
            e21 = input_ids2.index(30524); e22 = input_ids2.index(30525)
            if e21 <= e11:
                continue

            token2_sub = input_ids2[max(e12+1, e21-L): min(e22+L+1, len(input_ids2))]

            input_ids = [101] + input_ids1_sub + token2_sub + [102]
            aca_data.append({
                'input_ids': input_ids,
                'h_index': input_ids.index(30522),
                't_index': input_ids.index(30524),
                'label': rel_id,
                'relation': data1['relation'] + '-' + data2['relation']
            })

            for index in [30522, 30523, 30524, 30525]:
                assert index in input_ids and input_ids.count(index) == 1
                
        rel_id += 1

    for i in range(len(current_relations)):
        if current_relations[i] in ['P26', 'P3373', 'per:siblings', 'org:alternate_names', 'per:spous', 'per:alternate_names', 'per:other_family']:
            continue

        for data in rel_datas[i]:
            input_ids = data['input_ids']
            e11 = input_ids.index(30522); e12 = input_ids.index(30523)
            e21 = input_ids.index(30524); e22 = input_ids.index(30525)
            input_ids[e11] = 30524; input_ids[e12] = 30525
            input_ids[e21] = 30522; input_ids[e22] = 30523

            aca_data.append({
                'input_ids': input_ids,
                'h_index': input_ids.index(30522),
                't_index': input_ids.index(30524),
                'label': rel_id,
                'relation': data1['relation'] + '-reverse'
            })

            for index in [30522, 30523, 30524, 30525]:
                assert index in input_ids and input_ids.count(index) == 1
        rel_id += 1
    return aca_data

