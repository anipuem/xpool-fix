from abc import abstractmethod, ABC
# *类是从一堆*对象中抽取相同的内容而来的，那么*抽象类就是从一堆*类中抽取相同的内容而来的，内容包括数据属性和函数属性
# 通过：
#   1. 所在的 class 继承 abc.ABC
#   2. 给需要抽象的实例方法添加装饰器 @abstractmethod
# class 就变成了抽象类, 不能被直接实例化, 要想使用抽象类, 必须继承该类，并实现(重写)该类的所有抽象方法(函数)


class Config(ABC):  # 抽象类Config是用来继承的，而不是用来实例化的
    def __init__(self):
        args = self.parse_args()  # 调用类的parse_args()函数
        
        self.dataset_name = args.dataset_name
        self.videos_dir = args.videos_dir
        self.msrvtt_train_file = args.msrvtt_train_file
        self.num_frames = args.num_frames
        self.video_sample_type = args.video_sample_type
        self.input_res = args.input_res

        self.exp_name = args.exp_name
        self.model_path = args.model_path 
        self.output_dir = args.output_dir
        self.save_every = args.save_every
        self.log_step = args.log_step
        self.evals_per_epoch = args.evals_per_epoch
        self.load_epoch = args.load_epoch
        self.eval_window_size = args.eval_window_size
        self.metric = args.metric

        self.huggingface = args.huggingface
        self.arch = args.arch
        self.clip_arch = args.clip_arch
        self.embed_dim = args.embed_dim

        self.loss = args.loss
        self.clip_lr = args.clip_lr
        self.noclip_lr = args.noclip_lr
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.weight_decay = args.weight_decay
        self.warmup_proportion = args.warmup_proportion
    
        self.pooling_type = args.pooling_type
        self.k = args.k
        self.attention_temperature = args.attention_temperature
        self.num_mha_heads = args.num_mha_heads
        self.transformer_dropout = args.transformer_dropout

        self.num_workers = args.num_workers
        self.seed = args.seed
        self.no_tensorboard = args.no_tensorboard
        self.tb_log_dir = args.tb_log_dir

    # 通过@abstractmethod定义抽象方法，而无需实现其功能
    @abstractmethod
    def parse_args(self):
        raise NotImplementedError

