class MLMConfig:
    def __init__(self):
        self.warmup_steps = None
        self.save_path = None
        self.from_path = None
        self.device = None
        self.learning_rate = None
        self.weight_decay = None
        self.epochs = None
        self.batch_size = None
        self.prob_keep_ori = None
        self.prob_replace_rand = None
        self.mlm_probability = None
        self.special_tokens_mask = None
        self.prob_replace_mask = None

    def mlm_config(
            self,
            mlm_probability=0.15,
            special_tokens_mask=None,
            prob_replace_mask=0.8,
            prob_replace_rand=0.1,
            prob_keep_ori=0.1,
    ):
        self.mlm_probability = mlm_probability
        self.special_tokens_mask = special_tokens_mask
        self.prob_replace_mask = prob_replace_mask
        self.prob_replace_rand = prob_replace_rand
        self.prob_keep_ori = prob_keep_ori

    def train_config(
            self,
            batch_size,
            epochs,
            learning_rate,
            weight_decay,
            device,
            warmup_steps,
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.warmup_steps = warmup_steps

    def io_config(
            self,
            from_path,
            save_path,
    ):
        self.from_path = from_path
        self.save_path = save_path


def get_MLMConfig():
    config = MLMConfig()
    config.mlm_config()
    config.train_config(
        batch_size=4, epochs=10, learning_rate=1e-5, weight_decay=0,
        device='cuda', warmup_steps=500
    )
    config.io_config(
        from_path='xlm-roberta-base',
        save_path='output/MLM'
    )
    return config
