import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivergenceLoss(nn.Module):
    def forward(self, mu, logvar):
        ############################ Your code here ############################
        # TODO: compute the KL divergence loss for q(z|x) and p(z)
        # TODO: q(z|x) ~ N(mu, exp(logvar)), p(z) ~ N(0, 1)
        ########################################################################
        kl_divergence = None
        ########################################################################
        return kl_divergence.mean()


class ReconstructionLoss(nn.Module):
    def forward(self, x, x_recon):
        return F.mse_loss(x_recon, x)


class GANLossD(nn.Module):
    def forward(self, real, fake):
        ############################ Your code here ############################
        # TODO: compute the Hinge GAN loss for the discriminator
        ########################################################################
        loss = None
        ########################################################################
        return loss.mean()


class GANLossG(nn.Module):
    def forward(self, fake):
        ############################ Your code here ############################
        # TODO: compute the GAN loss for the generator
        ########################################################################
        loss = None
        ########################################################################
        return loss.mean()
