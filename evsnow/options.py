import argparse

def parse_options():
    """docstring for training configuration"""
    parser = argparse.ArgumentParser(description='Image desnowing using events')

    # args for arch selection
    parser.add_argument('--arch', type=str, default ='EvSnowNet_Paper',  help='architecture (only EvSnowNet_Paper is supported)')

    # args for training
    parser.add_argument('--train_dir', type=str, default ='/path/to/train/data',  help='dir of train data')
    parser.add_argument('--test_dir', type=str, default ='/path/to/test/data',  help='dir of test data')
    parser.add_argument('--train_ps', type=int, default=256, help='patch size of training sample')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--train_workers', type=int, default=8, help='train_dataloader workers')
    parser.add_argument('--gpu', type=str, default='0', help='GPUs')

    parser.add_argument('--nepoch', type=int, default=250, help='training epochs')
    parser.add_argument('--optimizer', type=str, default ='adamw', help='optimizer for training')
    parser.add_argument('--lr_initial', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay of')        
    parser.add_argument('--warmup', action='store_true', default=False, help='warmup') 
    parser.add_argument('--warmup_epochs', type=int,default=3, help='epochs for warmup')

    # args for logging
    parser.add_argument('--save_dir', type=str, default ='./logs/',  help='save dir')
    parser.add_argument('--save_images', action='store_true',default=False)
    parser.add_argument('--env', type=str, default ='_',  help='env')
    parser.add_argument('--checkpoint', type=int, default=10, help='epochs to save checkpoint')

    # args for resuming training
    parser.add_argument('--resume', action='store_true',default=False)
    parser.add_argument('--pretrain_weights',type=str, default='./log/Uformer_B/models/model_best.pth', help='path of pretrained_weights')

    return parser
