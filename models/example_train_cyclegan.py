import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import sys
import datetime
from cyclegan import CycleGAN
from generators import Generator_ResNet
from discriminators import ResDiscriminator
sys.path.insert(1, '../')
from utils.dataset import GAN_PKL_DATASET

BATCH_SIZE = 128
DATASET_PATH = "../processing/DATASET..." # Add dataset path
LOG_PATH = "./logs/"
NUM_WORKERS = 20
EPOCHS = 100
SAVE_DIRECTORY = str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))

def main():
    # Create the pytorch dataset and dataloader.
    dataset_train = GAN_PKL_DATASET(DATASET_PATH, ['Classic', 'Jazz'], normalise=False, on_off=True)

    dataset_val = GAN_PKL_DATASET(DATASET_PATH, ['Classic', 'Jazz'], normalise=False, on_off=True, num_segments=1)

    dataloader_train = DataLoader(dataset_train,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS)

    # Save one sample of each genre transfer every epoch
    dataloader_val = DataLoader(dataset_val,
                                batch_size=1,
                                shuffle=False,
                                num_workers=NUM_WORKERS)

    item = next(iter(dataloader_train))

    # Create cyclegan module
    cycle_gan = CycleGAN(generator=Generator_ResNet,
                         discriminator=ResDiscriminator,
                         save_directory=SAVE_DIRECTORY,
                         resolution=4,
                         gen_num_channels=64,
                         disc_num_channels=64,
                         input_shape=item[0].shape,
                         lr=0.000333,
                         lmd=13,
                         use_T = True,
                         use_C = True,
                         T_A_param = 0.5,
                         T_B_param = 0.5,
                         C_A_param = 0.5,
                         C_B_param = 0.5,
                         tolerance = 0.2
                        )

    if not os.path.isdir(os.path.join(SAVE_DIRECTORY, 'checkpoints')):
        os.mkdir(os.path.join(SAVE_DIRECTORY, 'checkpoints'))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(SAVE_DIRECTORY, 'checkpoints'))

    # Create trainer then train the model
    print('\nLogs are stored in "{}".'.format(LOG_PATH))
    trainer = pl.Trainer(
            max_epochs=EPOCHS,
            gpus=-1, # -1 will select all available gpus
            logger=pl.loggers.TensorBoardLogger(save_dir=LOG_PATH),
            log_every_n_steps=1,
            strategy='fsdp',
            default_root_dir = os.path.join(SAVE_DIRECTORY, 'checkpoints'),
            callbacks = [checkpoint_callback]
        )

    print('\nTraining...')
    trainer.fit(
        cycle_gan,
        train_dataloaders=dataloader_train,
        val_dataloaders=dataloader_val,
    )

    torch.save(cycle_gan.state_dict(), 'trained_models/cyclegan_timbre(BA)_comp(AB_BA).pt')

if __name__ == '__main__':
    main()