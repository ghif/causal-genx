from __future__ import annotations

import os

from runtime import configure_backend_from_argv

configure_backend_from_argv()

from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


TORCH_CONV_INIT = nnx.initializers.variance_scaling(1.0 / 3.0, "fan_in", "uniform")


def gaussian_kl(q_loc, q_logscale, p_loc, p_logscale):
    return -0.5 + p_logscale - q_logscale + 0.5 * (
        jnp.exp(q_logscale) ** 2 + (q_loc - p_loc) ** 2
    ) / (jnp.exp(p_logscale) ** 2)


def sample_gaussian(rng, loc, logscale):
    return loc + jnp.exp(logscale) * jax.random.normal(rng, loc.shape)


def _upsample(x, res):
    return jax.image.resize(x, (x.shape[0], res, res, x.shape[-1]), method="nearest")


def _avg_pool2d(x, kernel_size: int, stride: int):
    pooled = jax.lax.reduce_window(
        x,
        0.0,
        jax.lax.add,
        window_dimensions=(1, kernel_size, kernel_size, 1),
        window_strides=(1, stride, stride, 1),
        padding="VALID",
    )
    return pooled / float(kernel_size * kernel_size)


def _last_conv(block):
    return block.conv2 if block.version == "light" else block.conv4


def _nnx_list(values):
    list_cls = getattr(nnx, "List", None)
    return list_cls(values) if list_cls is not None else values


def _nnx_dict():
    dict_cls = getattr(nnx, "Dict", None)
    return dict_cls() if dict_cls is not None else {}


class Block(nnx.Module):
    def __init__(
        self,
        in_width: int,
        bottleneck: int,
        out_width: int,
        kernel_size: int = 3,
        residual: bool = True,
        down_rate: Optional[int] = None,
        version: Optional[str] = None,
        rngs: Optional[nnx.Rngs] = None,
    ):
        self.in_width = in_width
        self.bottleneck = bottleneck
        self.out_width = out_width
        self.kernel_size = kernel_size
        self.residual = residual
        self.down_rate = down_rate
        self.version = version
        padding = "VALID" if self.kernel_size == 1 else "SAME"
        if self.version == "light":
            self.conv1 = nnx.Conv(
                in_features=self.in_width,
                out_features=self.bottleneck,
                kernel_size=(self.kernel_size, self.kernel_size),
                padding=padding,
                kernel_init=TORCH_CONV_INIT,
                bias_init=nnx.initializers.zeros,
                rngs=rngs,
            )
            self.conv2 = nnx.Conv(
                in_features=self.bottleneck,
                out_features=self.out_width,
                kernel_size=(self.kernel_size, self.kernel_size),
                padding=padding,
                kernel_init=TORCH_CONV_INIT,
                bias_init=nnx.initializers.zeros,
                rngs=rngs,
            )
        else:
            self.conv1 = nnx.Conv(
                in_features=self.in_width,
                out_features=self.bottleneck,
                kernel_size=(1, 1),
                padding="VALID",
                kernel_init=TORCH_CONV_INIT,
                bias_init=nnx.initializers.zeros,
                rngs=rngs,
            )
            self.conv2 = nnx.Conv(
                in_features=self.bottleneck,
                out_features=self.bottleneck,
                kernel_size=(self.kernel_size, self.kernel_size),
                padding=padding,
                kernel_init=TORCH_CONV_INIT,
                bias_init=nnx.initializers.zeros,
                rngs=rngs,
            )
            self.conv3 = nnx.Conv(
                in_features=self.bottleneck,
                out_features=self.bottleneck,
                kernel_size=(self.kernel_size, self.kernel_size),
                padding=padding,
                kernel_init=TORCH_CONV_INIT,
                bias_init=nnx.initializers.zeros,
                rngs=rngs,
            )
            self.conv4 = nnx.Conv(
                in_features=self.bottleneck,
                out_features=self.out_width,
                kernel_size=(1, 1),
                padding="VALID",
                kernel_init=TORCH_CONV_INIT,
                bias_init=nnx.initializers.zeros,
                rngs=rngs,
            )
        if self.residual and self.in_width != self.out_width:
            self.res_proj = nnx.Conv(
                in_features=self.in_width,
                out_features=self.out_width,
                kernel_size=(1, 1),
                padding="VALID",
                kernel_init=TORCH_CONV_INIT,
                bias_init=nnx.initializers.zeros,
                rngs=rngs,
            )

    def __call__(self, x):
        h = x
        padding = "VALID" if self.kernel_size == 1 else "SAME"
        if self.version == "light":
            h = nnx.relu(h)
            h = self.conv1(h)
            h = nnx.relu(h)
            h = self.conv2(h)
        else:
            h = jax.nn.gelu(h)
            h = self.conv1(h)
            h = jax.nn.gelu(h)
            h = self.conv2(h)
            h = jax.nn.gelu(h)
            h = self.conv3(h)
            h = jax.nn.gelu(h)
            h = self.conv4(h)
        if self.residual:
            res_proj = getattr(self, "res_proj", None)
            if res_proj is not None:
                x = res_proj(x)
            h = x + h
        if self.down_rate:
            if isinstance(self.down_rate, float):
                res = max(1, int(h.shape[1] / self.down_rate))
                h = jax.image.resize(h, (h.shape[0], res, res, h.shape[-1]), method="linear")
            else:
                h = _avg_pool2d(h, int(self.down_rate), int(self.down_rate))
        return h


class Encoder(nnx.Module):
    def __init__(
        self,
        input_channels: int,
        input_res: int,
        enc_arch: str,
        widths: List[int],
        bottleneck: int = 4,
        vr: Optional[str] = None,
        rngs: Optional[nnx.Rngs] = None,
    ):
        self.input_channels = input_channels
        self.input_res = input_res
        self.enc_arch = enc_arch
        self.widths = widths
        self.bottleneck = bottleneck
        self.vr = vr
        self.stem = nnx.Conv(
            in_features=self.input_channels,
            out_features=self.widths[0],
            kernel_size=(7, 7),
            strides=(1, 1),
            padding="SAME",
            kernel_init=TORCH_CONV_INIT,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        stages = []
        for i, stage in enumerate(self.enc_arch.split(",")):
            start = stage.index("b") + 1
            end = stage.index("d") if "d" in stage else None
            n_blocks = int(stage[start:end])
            stages += [(self.widths[i], None) for _ in range(n_blocks)]
            if "d" in stage:
                stages += [(self.widths[i + 1], int(stage[stage.index("d") + 1 :]))]
        blocks = []
        for i, (width, d) in enumerate(stages):
            prev_width = stages[max(0, i - 1)][0]
            block_bottleneck = max(1, int(prev_width / self.bottleneck))
            blocks.append(Block(prev_width, block_bottleneck, width, down_rate=d, version=self.vr, rngs=rngs))
        scale = np.sqrt(1 / len(blocks))
        for block in blocks:
            last = _last_conv(block)
            last.kernel.value = last.kernel.value * scale
        self.blocks = _nnx_list(blocks)

    def __call__(self, x):
        x = self.stem(x)
        acts = {}
        for block in self.blocks:
            x = block(x)
            if x.shape[1] % 2 and x.shape[1] > 1:
                x = jnp.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0)))
            acts[x.shape[1]] = x
        return acts


class DecoderBlock(nnx.Module):
    def __init__(
        self,
        in_width: int,
        out_width: int,
        resolution: int,
        z_dim: int,
        context_dim: int,
        z_max_res: int = 192,
        bottleneck: int = 4,
        cond_prior: bool = False,
        q_correction: bool = True,
        vr: Optional[str] = None,
        rngs: Optional[nnx.Rngs] = None,
    ):
        self.in_width = in_width
        self.out_width = out_width
        self.resolution = resolution
        self.z_dim = z_dim
        self.context_dim = context_dim
        self.stochastic = self.resolution <= z_max_res
        self.bottleneck = bottleneck
        self.cond_prior = cond_prior
        self.q_correction = q_correction
        self.vr = vr
        k = 3 if self.resolution > 2 else 1
        self.prior = Block(
            self.in_width + (self.context_dim if self.cond_prior else 0),
            max(1, int(self.in_width / self.bottleneck)),
            2 * self.z_dim + self.in_width,
            kernel_size=k,
            residual=False,
            version=self.vr,
            rngs=rngs,
        )
        if self.stochastic:
            self.posterior = Block(
                2 * self.in_width + self.context_dim,
                max(1, int(self.in_width / self.bottleneck)),
                2 * self.z_dim,
                kernel_size=k,
                residual=False,
                version=self.vr,
                rngs=rngs,
            )
        self.z_proj = nnx.Conv(
            in_features=self.z_dim + self.context_dim,
            out_features=self.in_width,
            kernel_size=(1, 1),
            padding="VALID",
            kernel_init=TORCH_CONV_INIT,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        if not self.q_correction:
            self.z_feat_proj = nnx.Conv(
                in_features=self.z_dim + self.in_width,
                out_features=self.out_width,
                kernel_size=(1, 1),
                padding="VALID",
                kernel_init=TORCH_CONV_INIT,
                bias_init=nnx.initializers.zeros,
                rngs=rngs,
            )
        self.conv = Block(self.in_width, max(1, int(self.in_width / self.bottleneck)), self.out_width, kernel_size=k, version=self.vr, rngs=rngs)

    def forward_prior(self, z, pa=None, t=None):
        if self.cond_prior and pa is not None:
            z = jnp.concatenate([z, pa], axis=-1)
        z = self.prior(z)
        p_loc = z[..., : self.z_dim]
        p_logscale = z[..., self.z_dim : 2 * self.z_dim]
        p_features = z[..., 2 * self.z_dim :]
        if t is not None:
            p_logscale = p_logscale + jnp.log(t)
        return p_loc, p_logscale, p_features

    def forward_posterior(self, z, x, pa, t=None):
        h = jnp.concatenate([z, pa, x], axis=-1)
        q_loc, q_logscale = jnp.split(self.posterior(h), 2, axis=-1)
        if t is not None:
            q_logscale = q_logscale + jnp.log(t)
        return q_loc, q_logscale


class Decoder(nnx.Module):
    def __init__(
        self,
        widths: List[int],
        dec_arch: str,
        z_dim: int,
        context_dim: int,
        z_max_res: int = 192,
        bottleneck: int = 4,
        cond_prior: bool = False,
        q_correction: bool = True,
        bias_max_res: int = 192,
        vr: Optional[str] = None,
        hps: str = "morphomnist",
        rngs: Optional[nnx.Rngs] = None,
    ):
        self.widths = widths
        self.dec_arch = dec_arch
        self.z_dim = z_dim
        self.context_dim = context_dim
        self.z_max_res = z_max_res
        self.bottleneck = bottleneck
        self.cond_prior = cond_prior
        self.q_correction = q_correction
        self.bias_max_res = bias_max_res
        self.vr = vr
        self.hps = hps
        stages = []
        for i, stage in enumerate(self.dec_arch.split(",")):
            res = int(stage.split("b")[0])
            n_blocks = int(stage[stage.index("b") + 1 :])
            stages += [(res, self.widths[::-1][i]) for _ in range(n_blocks)]
        decoder_blocks = [
            DecoderBlock(
                width,
                stages[min(len(stages) - 1, i + 1)][1],
                res,
                self.z_dim,
                self.context_dim,
                z_max_res=self.z_max_res,
                bottleneck=self.bottleneck,
                cond_prior=self.cond_prior,
                q_correction=self.q_correction,
                vr=self.vr,
                rngs=rngs,
            )
            for i, (res, width) in enumerate(stages)
        ]
        self.blocks = _nnx_list(decoder_blocks)
        self.resolutions = tuple(sorted({r for r, _ in stages}))
        self.is_drop_cond = "morphomnist" in self.hps
        self.biases = _nnx_dict()
        for i, res in enumerate(self.resolutions):
            if res <= self.bias_max_res:
                width = self.widths[::-1][min(i, len(self.widths) - 1)]
                self.biases[str(res)] = nnx.Param(jnp.zeros((1, res, res, width), dtype=jnp.float32))
        self._scale_weights()

    def _scale_weights(self):
        scale = np.sqrt(1 / len(self.blocks))
        for block in self.blocks:
            block.z_proj.kernel.value = block.z_proj.kernel.value * scale
            _last_conv(block.conv).kernel.value = _last_conv(block.conv).kernel.value * scale
            _last_conv(block.prior).kernel.value = _last_conv(block.prior).kernel.value * 0.0

    def drop_cond(self, rng):
        options = jnp.array([[0, 1], [1, 0], [1, 1]], dtype=jnp.int32)
        opt = jax.random.randint(rng, (), 0, options.shape[0])
        return options[opt, 0], options[opt, 1]

    def __call__(self, parents, x=None, t=None, abduct=False, latents=None, rng=None, training=False):
        if rng is None:
            rng = jax.random.PRNGKey(0)
        bias = {}
        for i, res in enumerate(self.resolutions):
            if res <= self.bias_max_res:
                width = self.widths[::-1][min(i, len(self.widths) - 1)]
                bias[res] = self.biases[str(res)][...]
        h = z = bias[1].repeat(parents.shape[0], axis=0)
        p_sto, p_det = (1, 1)
        if training and self.cond_prior:
            p_sto, p_det = self.drop_cond(rng)
        stats = []
        keys = jax.random.split(rng, len(self.blocks) + 1)
        for i, block in enumerate(self.blocks):
            res = block.resolution
            pa = parents[:, :res, :res, :]
            if self.is_drop_cond:
                pa_sto = pa.at[..., 2:].multiply(p_sto)
                pa_det = pa.at[..., 2:].multiply(p_det)
            else:
                pa_sto = pa_det = pa
            if h.shape[1] < res:
                b = bias.get(res, 0)
                h = _upsample(h, res) + b
            if block.q_correction:
                p_input = h
            else:
                p_input = _upsample(z, res) if z.shape[1] < res else z
                p_input = p_input + bias.get(res, 0)
            p_loc, p_logscale, p_feat = block.forward_prior(p_input, pa_sto, t=t)
            posterior = getattr(block, "posterior", None)
            if posterior is not None and x is not None:
                q_loc, q_logscale = block.forward_posterior(h, x[res], pa, t=t)
                z = sample_gaussian(keys[i], q_loc, q_logscale)
                stat = {"kl": gaussian_kl(q_loc, q_logscale, p_loc, p_logscale)}
                if abduct:
                    if self.cond_prior:
                        stat["z"] = {"z": z, "q_loc": q_loc, "q_logscale": q_logscale}
                    else:
                        stat["z"] = z
                stats.append(stat)
            else:
                if latents is not None and i < len(latents) and latents[i] is not None:
                    z = latents[i]
                else:
                    z = sample_gaussian(keys[i], p_loc, p_logscale)
                    if abduct and self.cond_prior:
                        stats.append({"z": {"p_loc": p_loc, "p_logscale": p_logscale}})
            h = h + p_feat
            zpa = jnp.concatenate([z, pa], axis=-1)
            h = h + block.z_proj(zpa)
            h = block.conv(h)
            z_feat_proj = getattr(block, "z_feat_proj", None)
            if not block.q_correction and (i + 1) < len(self.blocks) and z_feat_proj is not None:
                z = z_feat_proj(jnp.concatenate([z, p_feat], axis=-1))
        return h, stats


class DGaussNet(nnx.Module):
    def __init__(self, input_channels: int, width: int, std_init: float = 0.1, rngs: Optional[nnx.Rngs] = None):
        self.input_channels = input_channels
        self.width = width
        self.std_init = std_init
        self.x_loc = nnx.Conv(
            in_features=self.width,
            out_features=self.input_channels,
            kernel_size=(1, 1),
            padding="VALID",
            kernel_init=TORCH_CONV_INIT,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.x_logscale = nnx.Conv(
            in_features=self.width,
            out_features=self.input_channels,
            kernel_size=(1, 1),
            padding="VALID",
            kernel_init=nnx.initializers.zeros if self.std_init > 0 else TORCH_CONV_INIT,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.channel_coeffs = (
            nnx.Conv(
                in_features=self.width,
                out_features=3,
                kernel_size=(1, 1),
                padding="VALID",
                kernel_init=TORCH_CONV_INIT,
                bias_init=nnx.initializers.zeros,
                rngs=rngs,
            )
            if self.input_channels == 3
            else None
        )

    def __call__(self, h, x=None, t=None):
        loc, logscale = self.x_loc(h), jnp.maximum(self.x_logscale(h), -9.0)
        if self.channel_coeffs is not None:
            coeff = jnp.tanh(self.channel_coeffs(h))
            if x is None:
                loc_red = jnp.clip(loc[..., 0], -1, 1)
                loc_green = jnp.clip(loc[..., 1] + coeff[..., 0] * loc_red, -1, 1)
                loc_blue = jnp.clip(loc[..., 2] + coeff[..., 1] * loc_red + coeff[..., 2] * loc_green, -1, 1)
            else:
                loc_red = loc[..., 0]
                loc_green = loc[..., 1] + coeff[..., 0] * x[..., 0]
                loc_blue = loc[..., 2] + coeff[..., 1] * x[..., 0] + coeff[..., 2] * x[..., 1]
            loc = jnp.stack([loc_red, loc_green, loc_blue], axis=-1)
        if t is not None:
            logscale = logscale + jnp.log(t)
        return loc, logscale

    def approx_cdf(self, x):
        return 0.5 * (1.0 + jnp.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x**3))))

    def nll(self, h, x):
        loc, logscale = self.__call__(h, x)
        centered_x = x - loc
        inv_stdv = jnp.exp(-logscale)
        plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
        cdf_plus = self.approx_cdf(plus_in)
        min_in = inv_stdv * (centered_x - 1.0 / 255.0)
        cdf_min = self.approx_cdf(min_in)
        log_cdf_plus = jnp.log(jnp.maximum(cdf_plus, 1e-12))
        log_one_minus_cdf_min = jnp.log(jnp.maximum(1.0 - cdf_min, 1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = jnp.where(
            x < -0.999,
            log_cdf_plus,
            jnp.where(x > 0.999, log_one_minus_cdf_min, jnp.log(jnp.maximum(cdf_delta, 1e-12))),
        )
        return -jnp.mean(log_probs, axis=(1, 2, 3))

    def sample(self, h, return_loc=True, t=None, rng=None):
        if return_loc:
            loc, logscale = self.__call__(h)
        else:
            loc, logscale = self.__call__(h, None, t=t)
            rng = rng or jax.random.PRNGKey(0)
            loc = loc + jnp.exp(logscale) * jax.random.normal(rng, loc.shape)
        return jnp.clip(loc, -1.0, 1.0), jnp.exp(logscale)


class HVAE(nnx.Module):
    def __init__(
        self,
        input_channels: int,
        input_res: int,
        enc_arch: str,
        dec_arch: str,
        widths: List[int],
        z_dim: int = 16,
        context_dim: int = 12,
        z_max_res: int = 192,
        bottleneck: int = 4,
        cond_prior: bool = False,
        q_correction: bool = False,
        bias_max_res: int = 64,
        x_like: str = "none_dgauss",
        kl_free_bits: float = 0.0,
        std_init: float = 0.1,
        hps: str = "morphomnist",
        vr: Optional[str] = None,
        rngs: Optional[nnx.Rngs] = None,
    ):
        self.input_channels = input_channels
        self.input_res = input_res
        self.enc_arch = enc_arch
        self.dec_arch = dec_arch
        self.widths = widths
        self.z_dim = z_dim
        self.context_dim = context_dim
        self.z_max_res = z_max_res
        self.bottleneck = bottleneck
        self.cond_prior = cond_prior
        self.q_correction = q_correction
        self.bias_max_res = bias_max_res
        self.x_like = x_like
        self.kl_free_bits = kl_free_bits
        self.std_init = std_init
        self.hps = hps
        self.vr = vr
        self.encoder = Encoder(self.input_channels, self.input_res, self.enc_arch, self.widths, self.bottleneck, self.vr, rngs=rngs)
        self.decoder = Decoder(
            self.widths,
            self.dec_arch,
            self.z_dim,
            self.context_dim,
            self.z_max_res,
            self.bottleneck,
            self.cond_prior,
            self.q_correction,
            self.bias_max_res,
            self.vr,
            self.hps,
            rngs=rngs,
        )
        self.likelihood = DGaussNet(self.input_channels, self.widths[0], self.std_init, rngs=rngs)

    def __call__(self, x, parents, beta=1.0, rng=None):
        acts = self.encoder(x)
        h, stats = self.decoder(parents=parents, x=acts, rng=rng, training=True)
        nll = self.likelihood.nll(h, x)
        if self.kl_free_bits > 0:
            fb = self.kl_free_bits
            kl = 0.0
            for stat in stats:
                kl = kl + jnp.maximum(
                    fb, jnp.sum(stat["kl"], axis=(1, 2)).mean(axis=0)
                ).sum()
        else:
            kl = 0.0
            for stat in stats:
                kl = kl + jnp.sum(stat["kl"], axis=(1, 2, 3)).mean()
        kl = kl / np.prod(x.shape[1:])
        nll = nll.mean()
        elbo = nll + beta * kl
        return {"elbo": elbo, "nll": nll, "kl": kl}

    def sample(self, parents, return_loc=True, t=None, rng=None):
        h, _ = self.decoder(parents=parents, t=t, rng=rng, training=False)
        return self.likelihood.sample(h, return_loc=return_loc, t=t, rng=rng)

    def abduct(self, x, parents, cf_parents=None, alpha=0.5, t=None, rng=None):
        acts = self.encoder(x)
        _, q_stats = self.decoder(parents=parents, x=acts, abduct=True, t=t, rng=rng, training=False)
        q_stats = [s["z"] for s in q_stats]
        if self.cond_prior and cf_parents is not None:
            _, p_stats = self.decoder(parents=cf_parents, abduct=True, t=t, rng=rng, training=False)
            p_stats = [s["z"] for s in p_stats]
            cf_zs = []
            for q_stat, p_stat in zip(q_stats, p_stats):
                q_loc = q_stat["q_loc"]
                q_scale = jnp.exp(q_stat["q_logscale"])
                u = (q_stat["z"] - q_loc) / q_scale
                p_loc = p_stat["p_loc"]
                p_var = jnp.exp(p_stat["p_logscale"]) ** 2
                r_loc = alpha * q_loc + (1.0 - alpha) * p_loc
                r_var = alpha**2 * q_scale**2 + (1.0 - alpha) ** 2 * p_var
                r_scale = jnp.sqrt(r_var)
                if t is not None:
                    r_scale = r_scale * t
                cf_zs.append(r_loc + r_scale * u)
            return cf_zs
        return q_stats

    def forward_latents(self, latents, parents, t=None, rng=None):
        h, _ = self.decoder(parents=parents, latents=latents, t=t, rng=rng, training=False)
        return self.likelihood.sample(h, t=t, rng=rng)


class SimpleVAE(HVAE):
    pass
