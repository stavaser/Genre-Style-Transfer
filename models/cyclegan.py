import os
import sys
sys.path.insert(1, '../')
from models.image_pool import ImagePool
from processing.postprocess import npy_to_midi
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

class CycleGAN(pl.LightningModule):
    """Module for training a CycleGAN.

    Attributes
    ----------
    generator : nn.Module
        Initilized Generator of choice.
    discriminator : nn.Module
        Initilized Discriminator of choice.
    save_directory : str
        Path to directory where samples should be saved.
    resolution : int
        Sample rate for the saved song. Should be at least the same as the denominator of the time signature.
    input_shape : int
        Shape of the input in the format of batch_size x channels x time x pitch_range.
    gen_num_channels : int
        Base number of channels, size of all layers gets calculated from this number.
    disc_num_channels : int
        Base number of channels, size of all layers gets calculated from this number.
    lr : float = 0.001
        Learning rate.
    b1 : float = 0.5
        Decay rate of first moment estimate for Adam optimiser.
    b2 : float = 0.999
        Decay rate of second moment estimate for Adam optimiser.
    lmd : int = 50
        Lambda constant. Multiply by cycle consistency loss and identity loss.
    use_T : bool = False
        Specify whether timbre discriminator should be used or not
    use_C : bool = False
        Specify whether composition discriminator should be used or not
    T_A_param : float = 0.38:
        Set the value to multiply generator_A timbre loss
    T_B_param : float = 0.38:
        Set the value to multiply generator_B timbre loss
    C_A_param : float = 0.38:
        Set the value to multiply generator_A composition loss
    C_B_param : float = 0.38:
        Set the value to multiply generator_B composition loss

    Methods
    -------
    init_weights : nn.Module
        Assign random values N(0, 0.02) to weights.
    """
    def __init__(self,
                 generator : nn.Module,
                 discriminator : nn.Module,
                 save_directory : str,
                 resolution : int,
                 input_shape : tuple,
                 gen_num_channels : int,
                 disc_num_channels : int,
                 lr : float = 0.001,
                 b1 : float = 0.5,
                 b2 : float = 0.999,
                 lmd : int = 50,
                 use_T : bool = False,
                 use_C : bool = False,
                 T_A_param: float = 0.38,
                 T_B_param: float = 0.38,
                 C_A_param: float = 0.38,
                 C_B_param: float = 0.38,
                 tolerance: float = 0.2
                ):
        super().__init__()

        if not os.path.isdir(save_directory):
            os.mkdir(save_directory)

        if not os.path.isdir(save_directory + '/generated_A'):
            os.mkdir(save_directory + '/generated_A')

        if not os.path.isdir(save_directory + '/generated_B'):
            os.mkdir(save_directory + '/generated_B')

        self.save_hyperparameters('lr', 'b1', 'b2', 'lmd', 'T_A_param', 'T_B_param',
                                  'C_A_param', 'C_B_param', 'tolerance')

        self.use_T = use_T
        self.use_C = use_C

        self.generator_A = generator(input_shape=input_shape, num_channels=gen_num_channels)
        self.generator_B = generator(input_shape=input_shape, num_channels=gen_num_channels)

        self.discriminator_A = discriminator(disc_num_channels, input_shape[1])
        self.discriminator_B = discriminator(disc_num_channels, input_shape[1])

        if self.use_T:
            self.discriminator_T = discriminator(disc_num_channels, input_shape[1], noise=False)
        if self.use_C:
            self.discriminator_C = discriminator(disc_num_channels, 1, noise=False)

        self.save_directory = save_directory
        self.resolution = resolution

        self.init_weights(self.generator_A)
        self.init_weights(self.generator_B)

        self.init_weights(self.discriminator_A)
        self.init_weights(self.discriminator_B)
        self.init_weights(self.discriminator_T)
        self.init_weights(self.discriminator_C)

        self.real_acc = torchmetrics.Accuracy()
        self.fake_acc = torchmetrics.Accuracy()

        self.fake_A_pool = ImagePool(50)  # create image buffer to store previously generated images
        self.fake_B_pool = ImagePool(50)  # create image buffer to store previously generated images

        self.counter = 0

    def forward(self, x, gen_choice: str):
        if gen_choice == 'A':
            return self.generator_A(x)
        if gen_choice == 'B':
            return self.generator_B(x)
        raise Exception('Wrong generator choice: {}'.format(gen_choice))

    def training_step(self, batch, batch_idx, optimizer_idx):
        segments_A, segments_B = batch

        if optimizer_idx == 0: # DISC A
            return self.disc_step(segments_A, segments_B, disc_choice='A')

        elif optimizer_idx == 1: # DISC B
            return self.disc_step(segments_A, segments_B, disc_choice='B')

        elif optimizer_idx == 2: # GEN A
            return self.gen_step(segments_A, segments_B, gen_choice='A')

        elif optimizer_idx == 3: # GEN B
            return self.gen_step(segments_A, segments_B, gen_choice='B')

        elif optimizer_idx == 4 and self.use_T: # DISC T
            return self.disc_step(segments_A, segments_B, disc_choice='T')

        elif optimizer_idx == 5 and self.use_C: # DISC C
            return self.disc_step(segments_A, segments_B, disc_choice='C')


    def gen_step(self, segments_A, segments_B, gen_choice='A'):
        # B->A
        if gen_choice=='A':
            # Adv loss
            fake_A = self(segments_B, 'A')
            pred_A = self.discriminator_A(fake_A)
            loss_adv = F.binary_cross_entropy(pred_A, torch.ones_like(pred_A))

            # Identity loss
            identity_A = self(segments_A, 'A')
            loss_identity = F.binary_cross_entropy(identity_A, segments_A)

            # Cycle loss
            rec_B = self(fake_A, 'B')
            loss_cycle = F.binary_cross_entropy(rec_B, segments_B)

            # Semi-total loss
            gen_loss = loss_adv + self.hparams.lmd*(loss_identity + loss_cycle)

            # Timbre loss
            if self.use_T:
                pred_T = self.discriminator_T(fake_A)
                loss_timbre = F.binary_cross_entropy(pred_T, torch.zeros_like(pred_T))
                gen_loss += self.hparams.T_A_param * loss_timbre

            # Composition loss
            if self.use_C:
                pred_C = self.discriminator_C(torch.amax(fake_A, dim=1).unsqueeze(1))
                loss_composition = F.binary_cross_entropy(pred_C, torch.zeros_like(pred_C))
                gen_loss += self.hparams.C_A_param * loss_composition

        # A->B
        if gen_choice=='B':
            # Adv loss
            fake_B = self(segments_A, 'B')
            pred_B = self.discriminator_B(fake_B)
            loss_adv = F.binary_cross_entropy(pred_B, torch.ones_like(pred_B))

            # Identity loss
            identity_B = self(segments_B, 'B')
            loss_identity = F.binary_cross_entropy(identity_B, segments_B)

            # Cycle loss
            rec_A = self(fake_B, 'A')
            loss_cycle = F.binary_cross_entropy(rec_A, segments_A)

            # Semi-total loss
            gen_loss = loss_adv + self.hparams.lmd*(loss_identity + loss_cycle)

            # Timbre loss
            if self.use_T:
                pred_T = self.discriminator_T(fake_B)
                loss_timbre = F.binary_cross_entropy(pred_T, torch.ones_like(pred_T))
                gen_loss += self.hparams.T_B_param * loss_timbre

            # Composition loss
            if self.use_C:
                pred_C = self.discriminator_C(torch.amax(fake_B, dim=1).unsqueeze(1))
                loss_composition = F.binary_cross_entropy(pred_C, torch.ones_like(pred_C))
                gen_loss += self.hparams.C_B_param * loss_composition

        self.log("Adv loss {}".format(gen_choice), loss_adv, on_step=True,
                 on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

        self.log("Identity loss {}".format(gen_choice), loss_identity, on_step=True,
                 on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

        self.log("Cycle loss {}".format(gen_choice), loss_cycle, on_step=True,
                 on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

        if self.use_T:
            self.log("Timbre loss {}".format(gen_choice), loss_timbre, on_step=True,
                    on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

        if self.use_C:
            self.log("Composition loss {}".format(gen_choice), loss_composition, on_step=True,
                        on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

        return gen_loss

    def disc_step(self, segments_A, segments_B, disc_choice='A'):

        addi_loss = None
        if disc_choice == 'A':
            fake_A = self(segments_B, 'A').detach()
            disc_real = self.discriminator_A(segments_A)
            disc_fake = self.discriminator_A(fake_A)
            fake_A = self.fake_A_pool.query(fake_A)
            disc_fake = self.discriminator_A(fake_A)

        elif disc_choice == 'B':
            fake_B = self(segments_A, 'B').detach()
            disc_real = self.discriminator_B(segments_B)
            disc_fake = self.discriminator_B(fake_B)
            fake_B = self.fake_B_pool.query(fake_B)
            disc_fake = self.discriminator_B(fake_B)

        elif disc_choice == 'T':
            fake_B = self(segments_A, 'B').detach()
            fake_A = self(segments_B, 'A').detach()
            disc_real = self.discriminator_T(segments_B)
            disc_fake = self.discriminator_T(segments_A)
            T_B = self.discriminator_T(fake_B)
            T_A = self.discriminator_T(fake_A)
            addi_loss = (F.binary_cross_entropy(T_B, torch.ones_like(T_B)) +
                           F.binary_cross_entropy(T_A, torch.zeros_like(T_A)))

        elif disc_choice == 'C':
            fake_B = self(segments_A, 'B').detach()
            fake_A = self(segments_B, 'A').detach()
            disc_real = self.discriminator_C(torch.amax(segments_B, dim=1).unsqueeze(1))
            disc_fake = self.discriminator_C(torch.amax(segments_A, dim=1).unsqueeze(1))
            C_B = self.discriminator_C(torch.amax(fake_B, dim=1).unsqueeze(1))
            C_A = self.discriminator_C(torch.amax(fake_A, dim=1).unsqueeze(1))
            addi_loss = (F.binary_cross_entropy(C_B, torch.ones_like(C_B)) +
                           F.binary_cross_entropy(C_A, torch.zeros_like(C_A)))


        else:
            raise Exception('Wrong disc choice: {}'.format(disc_choice))


        real_loss = F.binary_cross_entropy(disc_real, torch.ones_like(disc_real))
        fake_loss = F.binary_cross_entropy(disc_fake, torch.zeros_like(disc_fake))

        if addi_loss is None:
            disc_loss = real_loss + fake_loss
        else:
            disc_loss = real_loss + fake_loss + self.hparams.tolerance * addi_loss

        self.real_acc(torch.round(disc_real), torch.ones_like(disc_real, dtype=torch.int64))
        self.fake_acc(torch.round(disc_fake), torch.zeros_like(disc_fake, dtype=torch.int64))

        self.log("Discriminator loss {}".format(disc_choice), disc_loss, on_step=True, on_epoch=False,
                                                             prog_bar=False, logger=True)

        self.log("Discriminator accuracy real {}".format(disc_choice), self.real_acc, on_step=True, on_epoch=False,
                                                                      prog_bar=False, logger=True)

        self.log("Discriminator accuracy fake {}".format(disc_choice), self.fake_acc, on_step=True, on_epoch=False,
                                                                      prog_bar=False, logger=True)

        return disc_loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        # Optimisers

        # Discriminators
        optimiser_discriminator_A = torch.optim.Adam(self.discriminator_A.parameters(), lr=lr, betas=(b1, b2))
        optimiser_discriminator_B = torch.optim.Adam(self.discriminator_B.parameters(), lr=lr, betas=(b1, b2))

        # Generators
        optimiser_generator_A = torch.optim.Adam(self.generator_A.parameters(), lr=lr, betas=(b1, b2))
        optimiser_generator_B = torch.optim.Adam(self.generator_B.parameters(), lr=lr, betas=(b1, b2))

        # Extra discriminators
        optimiser_discriminator_T = torch.optim.Adam(self.discriminator_T.parameters(), lr=lr, betas=(b1, b2))
        optimiser_discriminator_C = torch.optim.Adam(self.discriminator_C.parameters(), lr=lr, betas=(b1, b2))

        optimisers = [optimiser_discriminator_A, optimiser_discriminator_B,
                      optimiser_generator_A, optimiser_generator_B,
                      optimiser_discriminator_T, optimiser_discriminator_C,
                      ]


        return optimisers, []

    def validation_step(self, batch, batch_idx):
        segments_A, segments_B = batch
        bpm_A, bpm_B = (120.0, 120.0)
        self._save_samples(segments_A, segments_B, bpm_A, bpm_B, batch_idx)
        self.counter += 1

    def _save_samples(self, segment_A, segment_B, bpm_A, bpm_B, idx):
        fake_B = self(segment_A, 'B').detach().round()
        rec_A = self(fake_B, 'A').detach().round()

        npy_to_midi(fake_B*127,
                    index=self.counter,
                    song_type='generated',
                    directory=self.save_directory + '/generated_B',
                    bpm=bpm_A,
                    resolution=self.resolution)
        npy_to_midi(rec_A*127,
                    index=self.counter,
                    song_type='cycle',
                    directory=self.save_directory + '/generated_B',
                    bpm=bpm_A,
                    resolution=self.resolution)
        npy_to_midi(segment_A.detach()*127,
                    index=self.counter,
                    song_type='original',
                    directory=self.save_directory + '/generated_B',
                    bpm=bpm_A,
                    resolution=self.resolution)

        fake_A = self(segment_B, 'A').detach().round()
        rec_B = self(fake_A, 'B').detach().round()

        npy_to_midi(fake_A*127,
                    index=self.counter,
                    song_type='generated',
                    directory=self.save_directory + '/generated_A',
                    bpm=bpm_B,
                    resolution=self.resolution)
        npy_to_midi(rec_B*127,
                    index=self.counter,
                    song_type='cycle',
                    directory=self.save_directory + '/generated_A',
                    bpm=bpm_B,
                    resolution=self.resolution)
        npy_to_midi(segment_B.detach()*127,
                    index=self.counter,
                    song_type='original',
                    directory=self.save_directory + '/generated_A',
                    bpm=bpm_B,
                    resolution=self.resolution)

    def init_weights(self, model):
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, (nn.BatchNorm2d)):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
                torch.nn.init.constant_(m.bias, 0)

