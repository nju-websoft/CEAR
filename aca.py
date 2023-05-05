# encoding:utf-8
import os
import nni
import math
import time
import json
import torch
import argparse
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel
from nni.utils import merge_parameter

from model import BertEncoder, Classifier
from data_process import FewRelProcessor, tacredProcessor
from utils import collate_fn, save_checkpoint, get_prototypes, memory_select, set_random_seed, compute_cos_sim, get_aca_data, get_augmentative_data

default_print = "\033[0m"
blue_print = "\033[1;34;40m"
yellow_print = "\033[1;33;40m"
green_print = "\033[1;32;40m"


def do_train(args, tokenizer, processor):
    memory = []
    memory_len = []
    relations = []
    testset = []
    prev_encoder, prev_classifier = None, None
    taskdatas = processor.get()
    rel2id = processor.get_rel2id() # {"rel": id}
    task_acc, memory_acc = [], []
    prototypes = None

    for i in range(args.task_num):
        task = taskdatas[i]
        traindata, _, testdata = task['train'], task['val'], task['test']
        train_len = task['train_len']
        testset += testdata
        new_relations = task['relation']
        relations += new_relations
        args.seen_rel_num = len(relations)

        # print some info
        print(f"{yellow_print}Training task {i}, relation set {task['relation']}.{default_print}")

        # train and val on task data
        current_encoder = BertEncoder(args, tokenizer, encode_style=args.encode_style)
        current_classifier = Classifier(args, args.hidden_dim, 3*args.seen_rel_num, prev_classifier).to(args.device)

        if prev_encoder is not None:
            current_encoder.load_state_dict(prev_encoder.state_dict())
        aug_traindata = get_aca_data(args, traindata, train_len, task['relation'])
        current_encoder = train_val_task(args, current_encoder, current_classifier, aug_traindata, testdata, rel2id, train_len)
        current_classifier.incremental_learning(args.seen_rel_num)

        # memory select
        print(f'{blue_print}Selecting memory for task {i}...{default_print}')
        new_memory, new_memory_len = memory_select(args, current_encoder, traindata, train_len)
        memory += new_memory
        memory_len += new_memory_len

        # evaluate on task testdata
        current_prototypes, current_proto_features = get_prototypes(args, current_encoder, traindata, train_len)
        acc = evaluate(args, current_encoder, current_classifier, testdata, rel2id)
        print(f'{blue_print}Accuracy of task {i} is {acc}.{default_print}')
        task_acc.append(acc)

        # train and val on memory data
        if prev_encoder is not None:
            print(f'{blue_print}Training on memory...{default_print}')
            task_prototypes = torch.cat([task_prototypes, current_prototypes], dim=0)
            task_proto_features = torch.cat([task_proto_features, current_proto_features], dim=0)

            prototypes = torch.cat([prototypes, current_prototypes], dim=0)
            proto_features = torch.cat([proto_features, current_proto_features], dim=0)

            current_model = (current_encoder, current_classifier)
            prev_model = (prev_encoder, prev_classifier)
            aug_memory = get_augmentative_data(args, memory, memory_len)
            current_encoder = train_val_memory(args, current_model, prev_model, memory, aug_memory, testset, rel2id, memory_len, memory_len, prototypes, proto_features, task_prototypes, task_proto_features)
        else:
            print(f"{blue_print}Initial task, won't train on memory.{default_print}")

        # update prototype
        print(f'{blue_print}Updating prototypes...{default_print}')
        if prev_encoder is not None:
            prototypes_replay, proto_features_replay = get_prototypes(args, current_encoder, memory, memory_len)
            prototypes, proto_features = (1-args.beta)*task_prototypes + args.beta*prototypes_replay, (1-args.beta)*task_proto_features + args.beta*proto_features_replay
            prototypes = F.layer_norm(prototypes, [args.hidden_dim])
            proto_features = F.normalize(proto_features, p=2, dim=1)
        else:
            task_prototypes, task_proto_features = current_prototypes, current_proto_features
            prototypes, proto_features = current_prototypes, current_proto_features

        # test
        print(f'{blue_print}Evaluating...{default_print}')
        if prev_encoder is not None:
            acc = evaluate(args, current_encoder, current_classifier, testset, rel2id, proto_features)
        else:
            acc = evaluate(args, current_encoder, current_classifier, testset, rel2id)
        print(f'{green_print}Evaluate finished, final accuracy over task 0-{i} is {acc}.{default_print}')
        memory_acc.append(acc)

        # save checkpoint
        print(f'{blue_print}Saving checkpoint of task {i}...{default_print}')

        prev_encoder = current_encoder
        prev_classifier = current_classifier

        nni.report_intermediate_result(acc)

    return task_acc, memory_acc


def train_val_task(args, encoder, classifier, traindata, valdata, rel2id, train_len):
    dataloader = DataLoader(traindata, batch_size=args.train_batch_size, shuffle=True, collate_fn=args.collate_fn, drop_last=True)

    optimizer = AdamW([
        {'params': encoder.parameters(), 'lr': args.encoder_lr},
        {'params': classifier.parameters(), 'lr': args.classifier_lr}
        ], eps=args.adam_epsilon)

    best_acc = 0.0
    for epoch in range(args.epoch_num_task):
        encoder.train()
        classifier.train()
        for step, batch in enumerate(tqdm(dataloader)):
            inputs = {
                'input_ids': batch[0].to(args.device),
                'attention_mask': batch[1].to(args.device),
                'h_index': batch[2].to(args.device),
                't_index': batch[3].to(args.device),
            }
            hidden, _ = encoder(**inputs)

            inputs = {
                'hidden': hidden,
                'labels': batch[4].to(args.device)
            }
            loss, _ = classifier(**inputs)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

    acc = evaluate(args, encoder, classifier, valdata, rel2id)
    best_acc = max(acc, best_acc)
    print(f'Evaluate on epoch {epoch}, accuracy={acc}, best_accuracy={best_acc}')

    return encoder


def train_val_memory(args, model, prev_model, traindata, aug_traindata, testdata, rel2id, memory_len, aug_memory_len, prototypes, proto_features, task_prototypes, task_proto_features):
    enc, cls = model
    prev_enc, prev_cls = prev_model
    dataloader = DataLoader(aug_traindata, batch_size=args.train_batch_size, shuffle=True, collate_fn=args.collate_fn, drop_last=True)

    optimizer = AdamW([
        {'params': enc.parameters(), 'lr': args.encoder_lr},
        {'params': cls.parameters(), 'lr': args.classifier_lr}
        ], eps=args.adam_epsilon)

    prev_enc.eval()
    prev_cls.eval()
    best_acc = 0.0
    for epoch in range(args.epoch_num_memory):
        enc.train()
        cls.train()
        for step, batch in enumerate(tqdm(dataloader)):
            enc_inputs = {
                'input_ids': batch[0].to(args.device),
                'attention_mask': batch[1].to(args.device),
                'h_index': batch[2].to(args.device),
                't_index': batch[3].to(args.device),
            }
            hidden, feature = enc(**enc_inputs)
            with torch.no_grad():
                prev_hidden, prev_feature = prev_enc(**enc_inputs)

            labels = batch[4].to(args.device)
            cont_loss = contrastive_loss(args, feature, labels, prototypes, proto_features, prev_feature)
            cont_loss.backward(retain_graph=True)

            rep_loss = replay_loss(args, cls, prev_cls, hidden, feature, prev_hidden, prev_feature, labels, prototypes, proto_features)
            rep_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        if (epoch+1) % 10 == 0:
            acc = evaluate(args, enc, cls, testdata, rel2id, proto_features)
            best_acc = max(best_acc, acc)
            print(f'Evaluate testset on epoch {epoch}, accuracy={acc}, best_accuracy={best_acc}')
            nni.report_intermediate_result(acc)

            prototypes_replay, proto_features_replay = get_prototypes(args, enc, traindata, memory_len)
            prototypes, proto_features = (1-args.beta)*task_prototypes + args.beta*prototypes_replay, (1-args.beta)*task_proto_features + args.beta*proto_features_replay
            prototypes = F.layer_norm(prototypes, [args.hidden_dim])
            proto_features = F.normalize(proto_features, p=2, dim=1)

    return enc


def contrastive_loss(args, feature, labels, prototypes, proto_features=None, prev_feature=None):
    # supervised contrastive learning loss
    dot_div_temp = torch.mm(feature, proto_features.T) / args.cl_temp # [batch_size, rel_num]
    dot_div_temp_norm = dot_div_temp - 1.0 / args.cl_temp
    exp_dot_temp = torch.exp(dot_div_temp_norm) + 1e-8 # avoid log(0)

    mask = torch.zeros_like(exp_dot_temp).to(args.device)
    mask.scatter_(1, labels.unsqueeze(1), 1.0)
    cardinalities = torch.sum(mask, dim=1)

    log_prob = -torch.log(exp_dot_temp / torch.sum(exp_dot_temp, dim=1, keepdim=True))
    scloss_per_sample = torch.sum(log_prob*mask, dim=1) / cardinalities
    scloss = torch.mean(scloss_per_sample)
    
    # focal knowledge distillation loss
    if prev_feature is not None:
        with torch.no_grad():
            prev_proto_features = proto_features[:proto_features.shape[1]-args.relnum_per_task]
            prev_sim = F.softmax(torch.mm(feature, prev_proto_features.T) / args.cl_temp / args.kd_temp, dim=1)

            prob = F.softmax(torch.mm(feature, proto_features.T) / args.cl_temp / args.kd_temp, dim=1)
            focal_weight = 1.0 - torch.gather(prob, dim=1, index=labels.unsqueeze(1)).squeeze()
            focal_weight = focal_weight ** args.gamma

            target = F.softmax(torch.mm(prev_feature, prev_proto_features.T) / args.cl_temp, dim=1) # [batch_size, prev_rel_num]

        source = F.log_softmax(torch.mm(feature, prev_proto_features.T) / args.cl_temp, dim=1) # [batch_size, prev_rel_num]
        target = target * prev_sim + 1e-8
        fkdloss = torch.sum(-source * target, dim=1)
        fkdloss = torch.mean(fkdloss * focal_weight)
    else:
        fkdloss = 0.0
    
    # margin loss
    if proto_features is not None:
        with torch.no_grad():
            sim = torch.mm(feature, proto_features.T)
            neg_sim = torch.scatter(sim, 1, labels.unsqueeze(1), -10.0)
            neg_indices = torch.argmax(neg_sim, dim=1)
        
        pos_proto = proto_features[labels]
        neg_proto = proto_features[neg_indices]

        positive = torch.sum(feature * pos_proto, dim=1)
        negative = torch.sum(feature * neg_proto, dim=1)

        marginloss = torch.maximum(args.margin - positive + negative, torch.zeros_like(positive).to(args.device))
        marginloss = torch.mean(marginloss)
    else:
        marginloss = 0.0

    loss = scloss + args.cl_lambda*marginloss + args.kd_lambda2*fkdloss
    return loss


def replay_loss(args, cls, prev_cls, hidden, feature, prev_hidden, prev_feature, labels, prototypes=None, proto_features=None):
    # cross entropy
    celoss, logits = cls(hidden, labels)
    with torch.no_grad():
        prev_logits, = prev_cls(prev_hidden)

    if prototypes is None:
        index = prev_logits.shape[1]
        source = F.log_softmax(logits[:, :index], dim=1)
        target = F.softmax(prev_logits, dim=1) + 1e-8
        kdloss = F.kl_div(source, target)
    else:
        # focal knowledge distillation
        with torch.no_grad():
            sim = compute_cos_sim(hidden, prototypes)
            prev_sim = sim[:, :prev_logits.shape[1]] # [batch_size, prev_rel_num]
            prev_sim = F.softmax(prev_sim / args.kd_temp, dim=1)

            prob = F.softmax(logits, dim=1)
            focal_weight = 1.0 - torch.gather(prob, dim=1, index=labels.unsqueeze(1)).squeeze()
            focal_weight = focal_weight ** args.gamma

        source = logits.narrow(1, 0, prev_logits.shape[1])
        source = F.log_softmax(source, dim=1)
        target = F.softmax(prev_logits, dim=1)
        target = target * prev_sim + 1e-8
        kdloss = torch.sum(-source * target, dim=1)
        kdloss = torch.mean(kdloss * focal_weight)
    
    rep_loss = celoss + args.kd_lambda1*kdloss
    return rep_loss


def evaluate(args, model, classifier, valdata, rel2id, proto_features=None):
    model.eval()
    dataloader = DataLoader(valdata, batch_size=args.test_batch_size, collate_fn=collate_fn, drop_last=False)
    pred_labels, golden_labels = [], []

    for i, batch in enumerate(tqdm(dataloader)):
        inputs = {
            'input_ids': batch[0].to(args.device),
            'attention_mask': batch[1].to(args.device),
            'h_index': batch[2].to(args.device),
            't_index': batch[3].to(args.device),
        }

        with torch.no_grad():
            hidden, feature = model(**inputs)
            logits = classifier(hidden)[0]
            prob_cls = F.softmax(logits, dim=1)
            if proto_features is not None:
                logits = torch.mm(feature, proto_features.T) / args.cl_temp
                prob_ncm = F.softmax(logits, dim=1)
                final_prob = args.alpha*prob_cls + (1-args.alpha)*prob_ncm
            else:
                final_prob = prob_cls

        # get pred_labels
        pred_labels += torch.argmax(final_prob, dim=1).cpu().tolist()
        golden_labels += batch[4].tolist()

    pred_labels = torch.tensor(pred_labels, dtype=torch.long)
    golden_labels = torch.tensor(golden_labels, dtype=torch.long)
    acc = float(torch.sum(pred_labels==golden_labels).item()) / float(len(golden_labels))
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--checkpoint_dir", default="checkpoint", type=str)
    parser.add_argument("--dataset_name", default="FewRel", type=str)
    parser.add_argument("--cuda", default=True, type=bool)
    parser.add_argument("--cuda_device", default=3, type=int)

    parser.add_argument("--plm_name", default="bert-base-uncased", type=str)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--test_batch_size", default=128, type=int)
    parser.add_argument("--epoch_num_task", default=10, type=int, help="Max training epochs.")
    parser.add_argument("--epoch_num_memory", default=10, type=int, help="Max training epochs.")
    parser.add_argument("--hidden_dim", default=768 , type=int, help="Output dimension of encoder.")
    parser.add_argument("--feature_dim", default=64, type=int, help="Output dimension of projection head.")
    parser.add_argument("--encoder_lr", default=1e-5, type=float, help="The initial learning rate of encoder for AdamW.")
    parser.add_argument("--classifier_lr", default=1e-3, type=float, help="The initial learning rate of classifier for AdamW.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")

    parser.add_argument("--alpha", default=0.5, type=float, help="Bagging Hyperparameter.")
    parser.add_argument("--beta", default=0.5, type=float, help="Prototype weight.")
    parser.add_argument("--cl_temp", default=0.1, type=float, help="Temperature for contrastive learning.")
    parser.add_argument("--cl_lambda", default=0.5, type=float, help="Hyperparameter for contrastive learning.")
    parser.add_argument("--margin", default=0.1, type=float, help="Hyperparameter for margin loss.")
    parser.add_argument("--kd_temp", default=0.5, type=float, help="Temperature for knowledge distillation.")
    parser.add_argument("--kd_lambda1", default=1.1, type=float, help="Hyperparameter for knowledge distillation.")
    parser.add_argument("--kd_lambda2", default=0.5, type=float, help="Hyperparameter for knowledge distillation.")
    parser.add_argument("--gamma", default=1.25, type=float, help="Hyperparameter of focal loss.")
    parser.add_argument("--encode_style", default="emarker", type=str, help="Encode style of encoder.")

    parser.add_argument("--experiment_num", default=5, type=int)
    parser.add_argument("--seed", default=2022, type=int)
    parser.add_argument("--set_task_order", default=True, type=bool)
    parser.add_argument("--read_from_task_order", default=True, type=bool)
    parser.add_argument("--task_num", default=10, type=int)
    parser.add_argument("--memory_size", default=10, type=int, help="Memory size for each relation.")
    parser.add_argument("--early_stop_patient", default=10, type=int)

    args = parser.parse_args()

    if args.cuda:
        device = "cuda:"+str(args.cuda_device)
    else:
        device = "cpu"
    args.device = device

    args.collate_fn = collate_fn
    
    tuner_params = nni.get_next_parameter()
    args = merge_parameter(args, tuner_params)

    tokenizer = BertTokenizer.from_pretrained(args.plm_name, additional_special_tokens=["[E11]", "[E12]", "[E21]", "[E22]"])

    s = time.time()
    task_results, memory_results = [], []
    for i in range(args.experiment_num):
        set_random_seed(args)
        if args.dataset_name == "FewRel":
            processor = FewRelProcessor(args, tokenizer)
        else:
            processor = tacredProcessor(args, tokenizer)
        if args.set_task_order:
            processor.set_task_order("task_order.json", i)
        if args.read_from_task_order:
            processor.set_read_from_order(i)

        task_acc, memory_acc = do_train(args, tokenizer, processor)
        print(f'{green_print}Result of experiment {i}:')
        print(f'task acc: {task_acc}')
        print(f'memory acc: {memory_acc}')
        print(f'Average: {sum(memory_acc)/len(memory_acc)}{default_print}')
        task_results.append(task_acc)
        memory_results.append(memory_acc)
        # torch.cuda.empty_cache()
    e = time.time()

    task_results = torch.tensor(task_results, dtype=torch.float32)
    memory_results = torch.tensor(memory_results, dtype=torch.float32)
    print(f'All task result: {task_results.tolist()}')
    print(f'All memory result: {memory_results.tolist()}')

    task_results = torch.mean(task_results, dim=0).tolist()
    memory_results = torch.mean(memory_results, dim=0)
    final_average = torch.mean(memory_results).item()
    print(f'Final task result: {task_results}')
    print(f'Final memory result: {memory_results.tolist()}')
    print(f'Final average: {final_average}')
    print(f'Time cost: {e-s}s.')

    nni.report_final_result(final_average)