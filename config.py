'''
instead of enter hyper parameters every time,
I construct a config class here, 
To load the hyper parameter, you can use:
    
    from config import DefaultConfig
    config = DefaultConfig()
    print(config.device) # 'cpu' for default

if you want to change the default parameter,
using:

    config._parse({use_gpu: 'True'})
    print(config.device) # 'cuda' if you have gpu
'''
import warnings
import torch

class DefaultConfig:
    train_img_root = 'D:/leftImg8bit_trainvaltest/leftImg8bit/train'
    train_target_root = 'D:/gtFine_trainvaltest/gtFine/train'
    val_img_root = 'D:/leftImg8bit_trainvaltest/leftImg8bit/test'
    val_target_root = 'D:/gtFine_trainvaltest/gtFine/test'
    epoch_num = 100
    batch_size = 50
    learning_rate = 1e-4
    output_path = './pretrained_model/model'
    gpu_index = None
    pretrained_model = './SETR'
    use_gpu = False
    device = torch.device('cuda') if use_gpu else 'cpu'


    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

config = DefaultConfig()
