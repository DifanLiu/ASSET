import pytorch_lightning as pl


class CheckpointEveryEpoch(pl.Callback):
    def __init__(self, start_epoc, save_path,):
        self.start_epoc = start_epoc
        self.file_path = save_path

    def on_train_epoch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train epoch """
        epoch = trainer.current_epoch
        if epoch >= self.start_epoc and trainer.global_rank == 0:
            ckpt_path = f"{self.file_path}/e{epoch:06d}.ckpt"
            trainer.save_checkpoint(ckpt_path)
            print("%s saved" % ckpt_path)


class CheckpointEveryIterations(pl.Callback):
    def __init__(self, start_iters, save_path, save_interval):
        self.start_iters = start_iters
        self.file_path = save_path
        self.save_interval = save_interval
        
    def on_train_batch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        iters = trainer.global_step
        if iters > self.start_iters and trainer.global_rank == 0 and iters % self.save_interval == 0:
            iters_in_k = iters // 1000
            ckpt_path = f"{self.file_path}/i{iters_in_k:06d}k.ckpt"
            trainer.save_checkpoint(ckpt_path)
            print("%s saved" % ckpt_path)