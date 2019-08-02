import argparse
import os
from train import Trainer
import torch
import torch.backends.cudnn as cudnn
from data_utils import DataLoader
from siamese import RankNet
import torch.optim as optim
import random
import logging
from infer import Tester


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', type=str, default='')
    parser.add_argument('--test-dir', type=str, default='')
    parser.add_argument('--val-dir', type=str, default='')
    parser.add_argument('--all-dir', type=str,
                        default='./prep/all_query_res_nonsorted.pkl')
    parser.add_argument('--outf', type=str, default='./model1')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--max-sent-len', type=int, default=512,
                        help='max sent len')
    parser.add_argument('--eval-batch-size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--max-sent-count', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--grad-norm', type=float, default=10,
                        help='gradient clipping')
    parser.add_argument('--model-type', type=str, default='lstm',
                        help='type of the top level model: [lstm | gru]')
    parser.add_argument('--dim', type=int, default=128,
                        help='feature layer size')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers at the top level model')
    parser.add_argument('--dropout', type=float, default=0,
                        help='dropout probability.')
    parser.add_argument('--bidir', dest='bidir', action='store_true')
    parser.add_argument('--no-bidir', dest='bidir', action='store_false')
    parser.set_defaults(bidir=True)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--max-niter', type=int, default=1000000,
                        help='number of iters in total')
    parser.add_argument('--eval-niter', type=int, default=200,
                        help='number of iters at each evaluation cycle')
    parser.add_argument('--lr-decay-pat', type=int, default=10,
                        help='learning rate decay patience')
    parser.add_argument('--loss-type', type=str, default='siamese')
    parser.add_argument('--margin', type=float, default=1.)
    parser.add_argument('--enable-bert', dest='enable-bert', action='store_true')
    parser.add_argument('--no-enable-bert', dest='enable-bert', action='store_false')
    parser.set_defaults(enable_bert=False)
    parser.add_argument('--top-n', type=int, default=5,
                        help='top-n')
    parser.add_argument('--mode', type=str, default='train',
                        help=['train | debug | test | vis | vis2'])

    opt = parser.parse_args()

    opt.max_sent_count = None if opt.max_sent_count == 0 else opt.max_sent_count

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    #logging(opt.outf, "Random Seed: {}".format(opt.manualSeed))
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    return opt


def main(opt):
    cudnn.benchmark = True

    # logging
    if not os.path.exists(opt.outf):
        os.system(f'mkdir -p {opt.outf}')

    if opt.mode == 'train':
        logging.basicConfig(filename="{}/log".format(opt.outf),
                            format='%(asctime)s %(message)s',
                            filemode='w')
    else:
        logging.basicConfig(filename="{}/log.aux".format(opt.outf),
                            format='%(asctime)s %(message)s',
                            filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if opt.train_dir != '' and opt.test_dir != '' and opt.val_dir != '':
        dataloader = DataLoader(train_dir=opt.train_dir,
                                test_dir=opt.test_dir, val_dir=opt.val_dir,
                                max_sent_count=opt.max_sent_count,
                                enable_bert=opt.enable_bert,
                                logger=logger)
    elif opt.all_dir != '':
        dataloader = DataLoader(all_dir=opt.all_dir,
                                max_sent_count=opt.max_sent_count,
                                enable_bert=opt.enable_bert,
                                logger=logger)
    else:
        raise ValueError

    model = RankNet(len(dataloader.vocab), 128, loss_type=opt.loss_type,
                    margin=opt.margin, enable_bert=opt.enable_bert,
                    max_sent_len=opt.max_sent_len,
                    max_sent_count=opt.max_sent_count)
    model.to('cuda')
    logger.info(model)

    if opt.mode == 'train':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr,
                               betas=(0.5, 0.999))
        trainer = Trainer(opt, model, dataloader, optimizer, logger)
        try:
            trainer.launch()
        except KeyboardInterrupt:
            print('terminated by key')
        trainer.eval_and_adjust(None)
    elif opt.mode == 'vis':
        tester = Tester(opt, model, dataloader)  # TODO logger needs not be done
        tester.inference_example()
    elif opt.mode == 'vis2':
        tester = Tester(opt, model, dataloader)
        tester.inference_example_sent()


if __name__ == '__main__':
    opt = get_opts()
    main(opt)

