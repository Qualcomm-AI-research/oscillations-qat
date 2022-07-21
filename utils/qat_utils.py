#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.

import copy

import torch

from quantization.hijacker import QuantizationHijacker
from quantization.quantized_folded_bn import BNFusedHijacker
from utils.imagenet_dataloaders import ImageNetDataLoaders


class MethodPropagator:
    """convenience class to allow multiple optimizers or LR schedulers to be used as if it
    were one optimizer/scheduler."""

    def __init__(self, propagatables):
        self.propagatables = propagatables

    def __getattr__(self, item):
        if callable(getattr(self.propagatables[0], item)):

            def propagate_call(*args, **kwargs):
                for prop in self.propagatables:
                    getattr(prop, item)(*args, **kwargs)

            return propagate_call
        else:
            return getattr(self.propagatables[0], item)

    def __str__(self):
        result = ""
        for prop in self.propagatables:
            result += str(prop) + "\n"
        return result

    def __iter__(self):
        for i in self.propagatables:
            yield i

    def __contains__(self, item):
        return item in self.propagatables


def get_dataloaders_and_model(config, load_type="fp32", **qparams):
    dataloaders = ImageNetDataLoaders(
        config.base.images_dir,
        224,
        config.base.batch_size,
        config.base.num_workers,
        config.base.interpolation,
    )

    model = config.base.architecture(
        pretrained=config.base.pretrained,
        load_type=load_type,
        model_dir=config.base.model_dir,
        **qparams,
    )
    if config.base.cuda:
        model = model.cuda()

    return dataloaders, model


class CompositeLoss:
    def __init__(self, loss_dict):
        """
        Composite loss of N separate loss functions. All functions are summed up.

        Note, each loss function gets as argument (prediction, target), even though if it might not
        need it. Other data independent instances need to be provided directly to the loss function
        (e.g. the model/weights in case of a regularization term.

        """
        self.loss_dict = loss_dict

    def __call__(self, prediction, target, *args, **kwargs):
        total_loss = 0
        for loss_func in self.loss_dict.values():
            total_loss += loss_func(prediction, target, *args, **kwargs)
        return total_loss


class UpdateFreezingThreshold:
    def __init__(self, tracker_dict, decay_schedule):
        self.tracker_dict = tracker_dict
        self.decay_schedule = decay_schedule

    def __call__(self, engine):
        if engine.state.iteration < self.decay_schedule.decay_start:
            # Put it always to 0 for real warm-start
            new_threshold = 0
        else:
            new_threshold = self.decay_schedule(engine.state.iteration)

        # Update trackers with new threshold
        for name, tracker in self.tracker_dict.items():
            tracker.freeze_threshold = new_threshold
        # print('Set new freezing threshold', new_threshold)


class UpdateDampeningLossWeighting:
    def __init__(self, bin_reg_loss, decay_schedule):
        self.dampen_loss = bin_reg_loss
        self.decay_schedule = decay_schedule

    def __call__(self, engine):
        new_weighting = self.decay_schedule(engine.state.iteration)
        self.dampen_loss.weighting = new_weighting
        # print('Set new bin reg weighting', new_weighting)


class DampeningLoss:
    def __init__(self, model, weighting=1.0, aggregation="sum"):
        """
        Calculates the dampening loss for all weights in a given quantized model. It is
        expected that all quantized weights are in a Hijacker module.

        """
        self.model = model
        self.weighting = weighting
        self.aggregation = aggregation

    def __call__(self, *args, **kwargs):
        total_bin_loss = 0
        for name, module in self.model.named_modules():
            if isinstance(module, QuantizationHijacker):
                # FP32 weight tensor, potential folded but before quantization
                weight, _ = module.get_weight_bias()
                # The matching weight quantizer (not manager, direct quantizer class)
                quantizer = module.weight_quantizer.quantizer
                total_bin_loss += dampening_loss(weight, quantizer, self.aggregation)
        return total_bin_loss * self.weighting


def dampening_loss(w_fp, quantizer, aggregation="sum"):
    # L &= (s*w_{int} - w)^2
    # We also need to add clipping for both cases, we can do so by using the forward
    w_q = quantizer(w_fp, skip_tracking=True).detach()  # this is also clipped and our target
    # clamp w in FP32 domain to not change range learning (min(max) is needed for per-channel)
    w_fp_clip = torch.min(torch.max(w_fp, quantizer.x_min), quantizer.x_max)
    loss = (w_q - w_fp_clip) ** 2
    if aggregation == "sum":
        return loss.sum()
    elif aggregation == "mean":
        return loss.mean()
    elif aggregation == "kernel_mean":
        return loss.sum(0).mean()
    else:
        raise ValueError(f"Aggregation method '{aggregation}' not implemented.")


class ReestimateBNStats:
    def __init__(self, model, data_loader, num_batches=50):
        super().__init__()
        self.model = model
        self.data_loader = data_loader
        self.num_batches = num_batches

    def __call__(self, engine):
        print("-- Reestimate current BN statistics --")
        reestimate_BN_stats(self.model, self.data_loader, self.num_batches)


def reestimate_BN_stats(model, data_loader, num_batches=50, store_ema_stats=False):
    # We set BN momentum to 1 an use train mode
    # -> the running mean/var have the current batch statistics
    model.eval()
    org_momentum = {}
    for name, module in model.named_modules():
        if isinstance(module, BNFusedHijacker):
            org_momentum[name] = module.momentum
            module.momentum = 1.0
            module.running_mean_sum = torch.zeros_like(module.running_mean)
            module.running_var_sum = torch.zeros_like(module.running_var)
            # Set all BNFusedHijacker modules to train mode for but not its children
            module.training = True

            if store_ema_stats:
                # Save the original EMA, make sure they are in buffers so they end in the state dict
                if not hasattr(module, "running_mean_ema"):
                    module.register_buffer("running_mean_ema", copy.deepcopy(module.running_mean))
                    module.register_buffer("running_var_ema", copy.deepcopy(module.running_var))
                else:
                    module.running_mean_ema = copy.deepcopy(module.running_mean)
                    module.running_var_ema = copy.deepcopy(module.running_var)

    # Run data for estimation
    device = next(model.parameters()).device
    batch_count = 0
    with torch.no_grad():
        for x, y in data_loader:
            model(x.to(device))
            # We save the running mean/var to a buffer
            for name, module in model.named_modules():
                if isinstance(module, BNFusedHijacker):
                    module.running_mean_sum += module.running_mean
                    module.running_var_sum += module.running_var

            batch_count += 1
            if batch_count == num_batches:
                break
    # At the end we normalize the buffer and write it into the running mean/var
    for name, module in model.named_modules():
        if isinstance(module, BNFusedHijacker):
            module.running_mean = module.running_mean_sum / batch_count
            module.running_var = module.running_var_sum / batch_count
            # We reset the momentum in case it would be used anywhere else
            module.momentum = org_momentum[name]
    model.eval()
