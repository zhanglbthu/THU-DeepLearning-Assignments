import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        beta_start=1e-4,
        beta_end=2e-2,
        num_timesteps=1000,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps

        betas = np.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

    def q_sample(self, x_start, t, noise):
        ############################ Your code here ############################
        # TODO: sample from q(x_t | x_0) with given x_0 and noise
        # TODO: hint: use extract function
        ########################################################################
        return x_start
        ########################################################################


    def p_losses(self, denoise_fn, x_start, y, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        pred_noise = denoise_fn(x_noisy, y, t)
        loss = F.mse_loss(pred_noise, noise)

        return loss
    
    def forward(self, denoise_fn, x_start, y):
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device).long()
        return self.p_losses(denoise_fn, x_start, y, t)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        return posterior_mean, posterior_variance
    
    def predict_start_from_noise(self, x, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * noise
        )

    def p_mean_variance(self, denoise_fn, x, y, t):
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=denoise_fn(x, y, t)
        )

        model_mean, model_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, model_variance

    @torch.no_grad()
    def p_sample(self, denoise_fn, x, y, t):
        b = x.shape[0]
        model_mean, model_variance = self.p_mean_variance(
            denoise_fn=denoise_fn, x=x, y=y, t=t
        )
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * model_variance.sqrt() * noise

    @torch.no_grad()
    def sample(self, denoise_fn, shape, y):
        b = shape[0]
        ############################ Your code here ############################
        # TODO: sample from the model
        # TODO: initially x_T = N(0, 1)
        # TODO: iterative sampling from p(x_{t-1} | x_t) until t == 0
        ########################################################################
        img = torch.randn(shape, device=y.device)
        ########################################################################
        return img

