from torch.utils.data import Dataset
import torch
import numpy
import random

ENTITY_CNT = 40943
RELATION_CNT = 18
CLS = ENTITY_CNT + RELATION_CNT
SEP = ENTITY_CNT + RELATION_CNT + 1
PAD = ENTITY_CNT + RELATION_CNT + 2


def read_2id_file(file_path):
    content = []
    with open(file_path) as f:
        cnt = next(f)
        for line in f:
            item, item_id = line.strip('\n').split('\t')
            content.append([item, item_id])
    return cnt, content


class CollateFn:
    def __init__(self, device, negative_sample_size):
        self.device = device
        self.negative_sample_size = negative_sample_size

    def collate_fn(self, batch_input):
        attention_masks = []
        position_ids = []
        seq_ids = []
        negative_seq_ids = []
        negative_attention_masks = []
        negative_position_ids = []
        for instance in batch_input:
            triplet = instance['positive_triplet']
            head_sample = instance['positive_head_sample']
            tail_sample = instance['positive_tail_sample']
            negative_entities = instance['negative_entities']
            negative_head_samples = instance['negative_head_samples']
            negative_tail_samples = instance['negative_tail_samples']
            neg_seq_ids = []
            neg_attention_masks = []
            neg_position_ids = []
            seq = [CLS] + triplet + [SEP]
            for sample1 in head_sample:
                seq += sample1
            for sample2 in tail_sample:
                seq += sample2
            for i in range(self.negative_sample_size):
                neg_seq = [CLS] + triplet[:-1] + [negative_entities[i]] + [SEP]
                for sample1 in negative_head_samples[i]:
                    neg_seq += sample1
                for sample2 in negative_tail_samples[i]:
                    neg_seq += sample2
                neg_seq_ids.append(neg_seq)
                neg_attention_masks.append([1] * len(neg_seq))
                neg_position_ids.append(list(range(len(neg_seq))))
            attention_mask = [1] * len(seq)
            position_id = list(range(len(seq)))
            seq_ids.append(seq)
            negative_seq_ids.append(neg_seq_ids)
            attention_masks.append(attention_mask)
            negative_attention_masks.append(neg_attention_masks)
            position_ids.append(position_id)
            negative_position_ids.append(neg_position_ids)
        attention_masks = torch.LongTensor(attention_masks).to(self.device)
        position_ids = torch.LongTensor(position_ids).to(self.device)
        seq_ids = torch.LongTensor(seq_ids).to(self.device)
        negative_seq_ids = torch.LongTensor(negative_seq_ids).to(self.device)
        negative_attention_masks = torch.LongTensor(negative_attention_masks).to(self.device)
        negative_position_ids = torch.LongTensor(negative_position_ids).to(self.device)
        return {
            'positive': {
                'input_ids': seq_ids,
                'attention_mask': attention_masks,
                'position_id': position_ids
                },
            'negative': {
                'input_ids': negative_seq_ids,
                'attention_mask': negative_attention_masks,
                'position_id': negative_position_ids
            }
        }


class MyDataset(Dataset):
    def __init__(self, file_path, neighbor_sample_size=3, negative_sample_size=5):
        super(MyDataset, self).__init__()
        self.neighbor_sample_size = neighbor_sample_size
        self.negative_sample_size = negative_sample_size
        self.data = self.read_corpus(file_path)[1]
        self.adjacent_matrix_in, self.adjacent_matrix_out = self.construct_adjacent_matrix()
        self.sample_enhanced_data = self.get_enhanced_data()

    def __len__(self):
        return len(self.sample_enhanced_data)

    def __getitem__(self, item):
        return self.sample_enhanced_data[item]

    def sample_neighbors(self, head, tail):
        head_sample = []
        tail_sample = []
        if len(self.adjacent_matrix_out[head]) == 0:
            head_sample = [[PAD, PAD] for _ in range(self.neighbor_sample_size)]
        else:
            for i in range(self.neighbor_sample_size):
                target_idx = random.randint(0, len(self.adjacent_matrix_out[head]) - 1)
                head_sample.append(self.adjacent_matrix_out[head][target_idx])
        if len(self.adjacent_matrix_in[tail]) == 0:
            tail_sample = [[PAD, PAD] for _ in range(self.neighbor_sample_size)]
        else:
            for i in range(self.neighbor_sample_size):
                target_idx = random.randint(0, len(self.adjacent_matrix_in[tail]) - 1)
                tail_sample.append(self.adjacent_matrix_in[tail][target_idx])
        return head_sample, tail_sample

    def get_enhanced_data(self):
        """
        第一种sample方法：采样两个node周围的节点及其关系
        :return:
        """
        sample_enhanced_data = []
        for head, relation, tail in self.data:
            positive_head_sample, positive_tail_sample = self.sample_neighbors(head, tail)
            negative_entities = []
            negative_head_samples = []
            negative_tail_samples = []
            for i in range(self.negative_sample_size):
                negative_entities.append(random.randint(0, ENTITY_CNT - 1))
                negative_head_sample, negative_tail_sample = self.sample_neighbors(head, negative_entities[-1])
                negative_head_samples.append(negative_head_sample)
                negative_tail_samples.append(negative_tail_sample)

            sample_enhanced_data.append(
                {'positive_triplet': [head, relation, tail],
                 'positive_head_sample': positive_head_sample,
                 'positive_tail_sample': positive_tail_sample,
                 'negative_entities': negative_entities,
                 'negative_head_samples': negative_head_samples,
                 'negative_tail_samples': negative_tail_samples})
        return sample_enhanced_data

    def construct_adjacent_matrix(self):
        adjacent_matrix_in = [[] for _ in range(ENTITY_CNT)]
        adjacent_matrix_out = [[] for _ in range(ENTITY_CNT)]
        for head, relation, tail in self.data:
            adjacent_matrix_out[head].append([tail, relation])
            adjacent_matrix_in[tail].append([head, relation])
        return adjacent_matrix_in, adjacent_matrix_out

    @staticmethod
    def read_corpus(file_path):
        data = []
        with open(file_path) as f:
            cnt = next(f)
            for line in f:
                head, tail, relation = line.strip('\n').split(' ')
                data.append([int(head), int(relation) + ENTITY_CNT, int(tail)])
        return cnt, data


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = MyDataset(file_path='data/WN18/train2id.txt')
    loader = DataLoader(dataset=dataset,
                        collate_fn=CollateFn(device='cuda:0', negative_sample_size=5).collate_fn,
                        batch_size=2)
    for _ in loader:
        print(_)
        break

