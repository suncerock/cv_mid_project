class FCNConfig(object):
    def __init__(self):
        self.backbone= 'ResNet'  # 'ResNet', 'AlexNet', 'VGGNet'
        self.decoder= '32x'  # '8x', '16x', '32x'
        self.pretrained_backbone= True,
        self.dataset= 'Cityscapes'  # 'VOC' or 'Cityscapes'

        if self.dataset == 'VOC':
            self.num_classes = 21
            self.ignore_index = 255
        elif self.dataset == 'Cityscapes':
            self.num_classes = 19
            self.ignore_index = 255
            
        if self.backbone == 'ResNet':
            self.decoder_c = (512, 256, 128, self.num_classes)
        elif backbone == 'AlexNet':
            self.decoder_c = (256, 192, 64, self.num_classes)
        elif backbone == 'VGGNet':
            self.decoder_c = (512, 512, 256, self.num_classes)
            
    
    def show_config(self):
        print("Model: FCN")
        print("Backbone: ", self.backbone)
        print("Decoder: ", self.decoder)
        print("Dataset: ", self.dataset)
        

class SETRConfig(object):
    def __init__(self):
        self.dataset= 'Cityscapes'  # 'VOC' or 'Cityscapes'
        self.nhead = 2
        self.num_layers = 2
        self.embed_dim = 256
        
        if self.dataset == 'VOC':
            self.num_classes = 21
            self.ignore_index = 255
        elif self.dataset == 'Cityscapes':
            self.num_classes = 19
            self.ignore_index = 255
            
    def show_config(self):
        print("Model: SETR")
        print("Dataset: ", self.dataset)
        print("nhead = {:d}, num_layers = {:d}".format(self.nhead, self.num_layers))
        

class DeepLabConfig(object):
    def __init__(self):
        self.dataset= 'Cityscapes'  # 'VOC' or 'Cityscapes'
        self.version = '1'
        
        if self.dataset == 'VOC':
            self.num_classes = 21
            self.ignore_index = 255
        elif self.dataset == 'Cityscapes':
            self.num_classes = 19
            self.ignore_index = 255
            
    def show_config(self):
        print("Model: DeepLabV{:s}".format(self.version))
        print("Dataset: ", self.dataset)
        
        
class Config(object):
    def __init__(self):
        self.batch_size = 16
        self.val_batch_size = 16
        self.num_epoches = 100
        self.num_eval_batch = 8
        
        self.lr = 1e-3
        self.T_max = 150
        self.last_epoch = -1
        
        self.save_path = './pretrained_model/FCN_ResNet_32x_pretrained_backbone_Cityscapes'
        self.pretrained = None        
        self.model = 'FCN'
        
        if self.model == 'FCN':
            self.model_config = FCNConfig()
        elif self.model == 'SETR':
            self.model_config = SETRConfig()
        elif self.model == 'DeepLab':
            self.model_config = DeepLabConfig()
            
    def show_config(self):
        print("Batch size: ", self.batch_size)
        print("Epoch num: ", self.num_epoches)
        print("Learning rate: ", self.lr)
        print("Validation batch size: ", self.val_batch_size)
        print("Evaluation batch num: ", self.num_eval_batch)
