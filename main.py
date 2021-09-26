import argparse
from datamodule import MyDataset, CollateFn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from model import KGEncoder
import torch
import os


def evaluate(dev_iter, model, args):
    model.eval()
    for batch_input in dev_iter:
        break
    pass


def train(train_iter, dev_iter, model, optimizer, train_sampler, args):
    # start_time = time.time()
    model.train()
    flag = False
    total_batch = 0
    total_cnt = 0
    hit = 0
    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        for batch_input in train_iter:
            model.zero_grad()
            loss, h = model.train_step(batch_input)
            total_cnt += (1 + args.negative_sample_size) * args.batch_size
            hit += h
            loss.backward()
            optimizer.step()
        if epoch % args.dev_epochs == 0:
            print('epoch: {}, ACC: {}'.format(epoch, hit / total_cnt))
            hit = 0
            total_cnt = 0
            model.train()
        if flag:
            break


def train_proc(gpu, args):
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=args.world_size,
                            rank=gpu)
    device = 'cuda:' + str(gpu)
    print('using device:', device)
    torch.manual_seed(0)
    # torch.cuda.set_device(gpu)
    train_dataset = MyDataset(file_path=args.train_file,
                              neighbor_sample_size=args.neighbor_sample_size,
                              negative_sample_size=args.negative_sample_size,
                              purpose='train')
    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=gpu)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  collate_fn=CollateFn(device=device,
                                                       negative_sample_size=args.negative_sample_size,
                                                       purpose='train').get_collate_fn(),
                                  batch_size=args.batch_size,
                                  sampler=train_sampler)

    dev_dataset = MyDataset(file_path=args.valid_file,
                            neighbor_sample_size=args.neighbor_sample_size,
                            negative_sample_size=None,
                            purpose='valid')
    dev_sampler = DistributedSampler(dev_dataset, num_replicas=args.world_size, rank=gpu)
    dev_dataloader = DataLoader(dataset=dev_dataset,
                                collate_fn=CollateFn(device=device,
                                                     negative_sample_size=None,
                                                     purpose='valid').get_collate_fn(),
                                batch_size=args.batch_size,
                                sampler=dev_sampler)
    model = KGEncoder(
        num_head=args.num_head,
        num_layers=args.num_layer,
        embedding_size=args.embed_dim,
        seq_length=5 + 4 * args.neighbor_sample_size
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    train(train_iter=train_dataloader,
          dev_iter=dev_dataloader,
          model=model,
          optimizer=optimizer,
          train_sampler=train_sampler,
          args=args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_layer', type=int, default=1)
    parser.add_argument('--num_head', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--negative_sample_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--neighbor_sample_size', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--train_file', type=str, default='/data2/yqc/KGE/data/WN18/train2id.txt')
    parser.add_argument('--valid_file', type=str, default='/data2/yqc/KGE/data/WN18/valid2id.txt')
    parser.add_argument('--test_file', type=str, default='/data2/yqc/KGE/data/WN18/test2id.txt')
    args = parser.parse_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '2385'
    args.world_size = len(args.device.split(','))
    torch.multiprocessing.spawn(train_proc, nprocs=args.n_gpu, args=(args,))


if __name__ == '__main__':
    main()
