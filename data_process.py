import os
import json
import random
import argparse

from tqdm import tqdm
from transformers import BertTokenizer

class FewRelProcessor():
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

        self.task_num = args.task_num
        self.relation_num = 80
        self.relnum_per_task = int(self.relation_num/self.task_num)
        
        args.relation_num = self.relation_num
        args.relnum_per_task = self.relnum_per_task

        self.train_num = 420
        self.val_num = 140
        self.test_num = 140

        self.task_order = None
        self.read_from_order = False
        self.relations = []
        self.rel2id = {}
    
    def _init_rel2id_relations(self):
        with open(os.path.join(self.args.data_dir, 'FewRel', 'pid2name.json'), 'r', encoding='utf-8') as file:
            pid2name = json.loads(file.read())
        rel2id, relations = {}, []
        for key, _ in pid2name:
            rel2id[key] = len(relations)
            relations.append(key)
        return rel2id, relations
    
    def set_task_order(self, filename, index):
        with open(os.path.join(self.args.data_dir, 'FewRel', filename), 'r', encoding='utf-8') as file:
            self.task_order = json.loads(file.read())
        self.task_order = self.task_order[index]
        for i in range(10):
            self.relations += self.task_order["T"+str(i+1)]
        for rel in self.relations:
            self.rel2id[rel] = self.relations.index(rel)
    
    def set_read_from_order(self, index):
        self.read_from_order = True
        self.order_index = index

    def _read_data_from_order(self, filename):
        ret, length = [], []
        with open(os.path.join(self.args.data_dir, 'FewRel', f'Exp{self.order_index}', filename), 'r', encoding='utf-8') as file:
            data = json.loads(file.read())
        for key, value in data.items():
            for instance in value:
                if len(instance["tokens"]) < 256:
                    instance["input_ids"] = instance["tokens"][:instance["tokens"].index(0)]
                else:
                    instance["input_ids"] = instance["tokens"]
                instance["h_index"] = instance["input_ids"].index(30522)
                instance["t_index"] = instance["input_ids"].index(30524)
                instance["relation"] = key
                instance["label"] = self.rel2id[key]
                ret.append(instance)
            length.append(len(value))
        return ret, length

    def get(self):
        if self.read_from_order:
            taskdatas = []
            for i in range(self.task_num):
                relation = self.relations[i*self.relnum_per_task: (i+1)*self.relnum_per_task]
                train, train_len = self._read_data_from_order(f"task_{i}_train.json")
                val, _ = self._read_data_from_order(f"task_{i}_val.json")
                test, _ = self._read_data_from_order(f"task_{i}_test.json")
                taskdatas.append({
                    'relation': relation,
                    'train': train,
                    'val': val,
                    'test': test,
                    'train_len': train_len
                })
        else:
            data = self._read()
            taskdatas = self._divide(data)
        return taskdatas

    def get_rel2id(self):
        return self.rel2id

    def _read(self):
        """
        Read FewRel train and val data.
        """
        print("Reading dataset...")

        with open(os.path.join(self.args.data_dir, 'FewRel', 'train_wiki.json'), 'r', encoding='utf-8') as file:
            train_set = json.loads(file.read())
        with open(os.path.join(self.args.data_dir, 'FewRel', 'val_wiki.json'), 'r', encoding='utf-8') as file:
            val_set = json.loads(file.read())
        dataset = train_set
        dataset.update(val_set)

        print("Read finished!")
        return dataset

    def _divide(self, dataset):
        """
        Divide dataset into <self.task_num> tasks.
        """
        if self.task_order is None:
            relations = list(dataset.keys())
            random.shuffle(relations)
            for rel in relations:
                self.relations.append(rel)
                self.rel2id[rel] = self.relations.index(rel)
        relations = self.relations

        print(f'Dividing dataset into {self.task_num} tasks...')

        taskdatas = []
        for i in tqdm(range(self.task_num)):
            train, val, test = [], [], []
            train_len = []

            relation = relations[i*self.relnum_per_task: (i+1)*self.relnum_per_task]
            for r in relation:
                rdata = dataset[r]
                random.shuffle(rdata)
                for instance in rdata:
                    # todo add entity marker and index
                    instance['relation'] = r
                    instance['label'] = self.relations.index(r)
                    sentence = ' '.join(instance['tokens']).lower()
                    h = ' '.join(instance['tokens'][instance['h'][-1][0][0]: instance['h'][-1][0][-1]+1]).lower()
                    t = ' '.join(instance['tokens'][instance['t'][-1][0][0]: instance['t'][-1][0][-1]+1]).lower()
                    sentence = sentence.replace(h, f"[E11] {h} [E12]")
                    sentence = sentence.replace(t, f"[E21] {t} [E22]")
                    instance['input_ids'] = self.tokenizer.encode(sentence)
                    instance['h_index'] = instance['input_ids'].index(self.tokenizer.additional_special_tokens_ids[0])
                    instance['t_index'] = instance['input_ids'].index(self.tokenizer.additional_special_tokens_ids[2])
                train.extend(rdata[0: self.train_num])
                val.extend(rdata[self.train_num: self.train_num+self.val_num])
                test.extend(rdata[self.train_num+self.val_num:])
                train_len.append(self.train_num)

            task = {
                'relation': relation,
                'train': train, # {"tokens", "h", "t", "relation", "label", "input_ids", "h_index", "t_index"}
                'val': val,
                'test': test,
                'train_len': train_len
            }
            taskdatas.append(task)
        return taskdatas

    def _tokenize(self, taskdatas):
        for taskdata in taskdatas:
            for name in ["train", "val", "test"]:
                for sample in taskdata[name]:
                    input_ids = self.tokenizer.encode(' '.join(sample['tokens']))
                    sample['input_ids'] = input_ids
        return taskdatas


class tacredProcessor():
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

        self.task_num = args.task_num
        self.relation_num = 40
        self.relnum_per_task = int(self.relation_num/self.task_num)
        
        args.relation_num = self.relation_num
        args.relnum_per_task = self.relnum_per_task

        self.train_num = 320
        self.val_num = 0
        self.test_num = 40

        self.task_order = None
        self.read_from_order = False
        self.relations = []
        self.rel2id = {}
    
    def set_task_order(self, filename, index):
        with open(os.path.join(self.args.data_dir, 'tacred', filename), 'r', encoding='utf-8') as file:
            self.task_order = json.loads(file.read())
        self.task_order = self.task_order[index]
        for i in range(10):
            self.relations += self.task_order["T"+str(i+1)]
        for rel in self.relations:
            self.rel2id[rel] = self.relations.index(rel)
        print(f"Experiment Num {index}")
        assert len(self.relations) == len(self.rel2id)
    
    def set_read_from_order(self, index):
        self.read_from_order = True
        self.order_index = index

    def _read_data_from_order(self, filename):
        ret, length = [], []
        with open(os.path.join(self.args.data_dir, 'tacred', f'Exp{self.order_index}', filename), 'r', encoding='utf-8') as file:
            data = json.loads(file.read())
        for key, value in data.items():
            for instance in value:
                tail = instance["tokens"].index(0) if 0 in instance["tokens"] else 256
                instance["input_ids"] = instance["tokens"][:tail]
                instance["h_index"] = instance["input_ids"].index(30522)
                instance["t_index"] = instance["input_ids"].index(30524)
                instance["relation"] = key
                instance["label"] = self.rel2id[key]
                ret.append(instance)
            length.append(len(value))
        return ret, length

    def get(self):
        if self.read_from_order:
            taskdatas = []
            for i in range(self.task_num):
                relation = self.relations[i*self.relnum_per_task: (i+1)*self.relnum_per_task]
                train, train_len = self._read_data_from_order(f"task_{i}_train.json")
                val, _ = self._read_data_from_order(f"task_{i}_val.json")
                test, _ = self._read_data_from_order(f"task_{i}_test.json")
                taskdatas.append({
                    'relation': relation,
                    'train': train,
                    'val': val,
                    'test': test,
                    'train_len': train_len
                })
        else:
            data = self._read()
            taskdatas = self._divide(data)
        return taskdatas
    
    def get_rel2id(self):
        return self.rel2id
    
    def get_ralation_description(self):
        descriptions = [self.tokenizer.encode(' '.join(relation[relation.index(":")+1:].split('_'))) for relation in self.relations]
        return descriptions

    
    def _read(self):
        """
        Read tacred dataset.
        """
        print("Reading dataset...")
        with open(os.path.join(self.args.data_dir, 'tacred', 'train.json'), 'r', encoding='utf-8') as file:
            train_set = json.loads(file.read())
        with open(os.path.join(self.args.data_dir, 'tacred', 'dev.json'), 'r', encoding='utf-8') as file:
            val_set = json.loads(file.read())
        with open(os.path.join(self.args.data_dir, 'tacred', 'test.json'), 'r', encoding='utf-8') as file:
            test_set = json.loads(file.read())
        dataset = train_set
        dataset += val_set
        dataset += test_set
        if self.args.set_task_order:
            dataset = self._convert_to_fewrel_form(dataset)
        else:
            with open(os.path.join(self.args.data_dir, 'tacred', 'relations.json'), 'r', encoding='utf-8') as file:
                self.relations = json.loads(file.read())
            dataset = self._convert_to_fewrel_form(dataset)
            self.relations = []
        print("Read finished!")
        return dataset
    
    def _convert_to_fewrel_form(self, dataset):
        new_dataset = {}
        for sample in dataset:
            relation = sample["relation"]
            if relation not in self.relations:
                continue
            h = ' '.join(sample["token"][sample['subj_start']:sample['subj_end']+1])
            t = ' '.join(sample["token"][sample['obj_start']:sample['obj_end']+1])
            new_sample = {
                "tokens": sample["token"],
                "h": [h, "Q114514", [[i for i in range(sample['subj_start'], sample['subj_end']+1)]]],
                "t": [t, "Q114514", [[i for i in range(sample['obj_start'], sample['obj_end']+1)]]]
            }
            if relation not in new_dataset:
                new_dataset[relation] = [new_sample]
            else:
                new_dataset[relation].append(new_sample)
        return new_dataset
    
    def _divide(self, dataset):
        """
        Divide dataset into <self.task_num> tasks.
        """
        if self.task_order is None:
            relations = list(dataset.keys())
            random.shuffle(relations)
            for rel in relations:
                self.relations.append(rel)
                self.rel2id[rel] = self.relations.index(rel)
        relations = self.relations

        print(f'Dividing dataset into {self.task_num} tasks...')

        taskdatas = []
        for i in tqdm(range(self.task_num)):
            train, val, test = [], [], []
            train_len = []

            relation = relations[i*self.relnum_per_task: (i+1)*self.relnum_per_task]
            for r in relation:
                rdata = dataset[r]
                random.shuffle(rdata)
                for instance in rdata:
                    # todo add entity marker and index
                    instance['relation'] = r
                    instance['label'] = self.relations.index(r)
                    h_start = instance['h'][-1][0][0]
                    instance['tokens'].insert(h_start, "[E11]")
                    h_end = instance['h'][-1][0][-1]+2
                    instance['tokens'].insert(h_end, "[E12]")
                    t_start = instance['t'][-1][0][0]
                    t_end = instance['t'][-1][0][-1]+2
                    if h_start < t_start:
                        t_start += 2
                        t_end += 2
                    instance['tokens'].insert(t_start, "[E21]")
                    instance['tokens'].insert(t_end, "[E22]")
                    sentence = ' '.join(instance['tokens']).lower()
                    sentence = sentence.replace("[e11]", "[E11]")
                    sentence = sentence.replace("[e12]", "[E12]")
                    sentence = sentence.replace("[e21]", "[E21]")
                    sentence = sentence.replace("[e22]", "[E22]")
                    instance['input_ids'] = self.tokenizer.encode(sentence)
                    instance['h_index'] = instance['input_ids'].index(self.tokenizer.additional_special_tokens_ids[0])
                    instance['t_index'] = instance['input_ids'].index(self.tokenizer.additional_special_tokens_ids[2])
                test_count = 0
                train_count = 0
                for i in range(len(rdata)):
                    if i < len(rdata) // 5 and test_count <= self.test_num:
                        test.append(rdata[i])
                        test_count += 1
                    else:
                        train.append(rdata[i])
                        train_count += 1
                        if train_count >= self.train_num:
                            break
                train_len.append(train_count)
            task = {
                'relation': relation,
                'train': train, # {"tokens", "h", "t", "relation", "label", "input_ids", "h_index", "t_index"}
                'val': val,
                'test': test,
                'train_len': train_len
            }
            taskdatas.append(task)
        return taskdatas

