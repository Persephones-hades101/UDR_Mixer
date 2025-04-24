# finetune_pruned.py
import lightning.pytorch as pl
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset_utils import TrainDataset
from utils.schedulers import LinearWarmupCosineAnnealingLR
from options import options as opt               # ← same CLI object you already use
from net.model import UDRMixer

PRUNED_WEIGHTS = "udr_mixer_pruned.pth"          # output of step-1
EPOCHS_EXTRA   = 8                               # small recovery run
LR_WARM        = 1e-5
LR_MAIN        = 2e-4

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = UDRMixer(dim=64, upscaling_factor=2)
        self.net.load_state_dict(torch.load(PRUNED_WEIGHTS, map_location="cpu"))
        self.loss_fn = nn.L1Loss()

    def forward(self, x): return self.net(x)

    def training_step(self, batch, batch_idx):
        degr, clean = batch
        out  = self.net(degr)
        loss = self.loss_fn(out, clean)
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        opti = optim.AdamW(self.parameters(), lr=LR_WARM)
        sched = LinearWarmupCosineAnnealingLR(opti,
                                              warmup_epochs=1,
                                              max_epochs=EPOCHS_EXTRA)
        return [opti], [sched]

def main():
    trainset   = TrainDataset(opt)
    trainload  = DataLoader(trainset, batch_size=opt.batch_size,
                            shuffle=True, num_workers=opt.num_workers, drop_last=True)
    model  = Model()
    trainer = pl.Trainer(max_epochs=EPOCHS_EXTRA,
                         accelerator="gpu", devices=opt.num_gpus)
    trainer.fit(model, trainload)
    # save final
    trainer.save_checkpoint("udr_mixer_pruned_finetuned.ckpt")

if __name__ == "__main__":
    main()
