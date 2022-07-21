#!/usr/bin/env python
# Copyright (c) 2022 Qualcomm Technologies, Inc.
# All Rights Reserved.
import torch
from timm.models import create_model
from timm.models.efficientnet_blocks import DepthwiseSeparableConv, InvertedResidual
from torch import nn

from quantization.autoquant_utils import quantize_sequential, quantize_model
from quantization.base_quantized_classes import QuantizedActivation, FP32Acts
from quantization.base_quantized_model import QuantizedModel


class QuantizedInvertedResidual(QuantizedActivation):
    def __init__(self, inv_res_orig, **quant_params):
        super().__init__(**quant_params)

        assert inv_res_orig.drop_path_rate == 0.0
        assert isinstance(inv_res_orig.se, nn.Identity)

        self.has_residual = inv_res_orig.has_residual

        conv_pw = nn.Sequential(inv_res_orig.conv_pw, inv_res_orig.bn1, inv_res_orig.act1)
        self.conv_pw = quantize_sequential(conv_pw, **quant_params)[0]

        conv_dw = nn.Sequential(inv_res_orig.conv_dw, inv_res_orig.bn2, inv_res_orig.act2)
        self.conv_dw = quantize_sequential(conv_dw, **quant_params)  # [0]

        conv_pwl = nn.Sequential(inv_res_orig.conv_pwl, inv_res_orig.bn3)
        self.conv_pwl = quantize_sequential(conv_pwl, **quant_params)[0]

    def forward(self, x):
        residual = x
        # Point-wise expansion
        x = self.conv_pw(x)
        # Depth-wise convolution
        x = self.conv_dw(x)
        # Point-wise linear projection
        x = self.conv_pwl(x)

        if self.has_residual:
            x += residual
            x = self.quantize_activations(x)
        return x


class QuantizedDepthwiseSeparableConv(QuantizedActivation):
    def __init__(self, dws_orig, **quant_params):
        super().__init__(**quant_params)

        assert dws_orig.drop_path_rate == 0.0
        assert isinstance(dws_orig.se, nn.Identity)

        self.has_residual = dws_orig.has_residual

        conv_dw = nn.Sequential(dws_orig.conv_dw, dws_orig.bn1, dws_orig.act1)
        self.conv_dw = quantize_sequential(conv_dw, **quant_params)[0]

        conv_pw = nn.Sequential(dws_orig.conv_pw, dws_orig.bn2, dws_orig.act2)
        self.conv_pw = quantize_sequential(conv_pw, **quant_params)[0]

    def forward(self, x):
        residual = x
        # Depth-wise convolution
        x = self.conv_dw(x)
        # Point-wise projection
        x = self.conv_pw(x)
        if self.has_residual:
            x += residual
            x = self.quantize_activations(x)
        return x


class QuantizedEfficientNetLite(QuantizedModel):
    def __init__(self, base_model, input_size=(1, 3, 224, 224), quant_setup=None, **quant_params):
        super().__init__(input_size)

        specials = {
            InvertedResidual: QuantizedInvertedResidual,
            DepthwiseSeparableConv: QuantizedDepthwiseSeparableConv,
        }

        conv_stem = nn.Sequential(base_model.conv_stem, base_model.bn1, base_model.act1)
        self.conv_stem = quantize_model(conv_stem, specials=specials, **quant_params)[0]

        self.blocks = quantize_model(base_model.blocks, specials=specials, **quant_params)

        conv_head = nn.Sequential(base_model.conv_head, base_model.bn2, base_model.act2)
        self.conv_head = quantize_model(conv_head, specials=specials, **quant_params)[0]

        self.global_pool = base_model.global_pool

        base_model.classifier.__class__ = nn.Linear  # Small hack to work with autoquant
        self.classifier = quantize_model(base_model.classifier, **quant_params)

        if quant_setup == "FP_logits":
            print("Do not quantize output of FC layer")
            self.classifier.activation_quantizer = FP32Acts()  # no activation quantization of
            # logits
        elif quant_setup == "LSQ":
            print("Set quantization to LSQ (first+last layer in 8 bits)")
            # Weights of the first layer
            self.conv_stem.weight_quantizer.quantizer.n_bits = 8
            # The quantizer of the last conv_layer layer (input to global)
            self.conv_head.activation_quantizer.quantizer.n_bits = 8
            # Weights of the last layer
            self.classifier.weight_quantizer.quantizer.n_bits = 8
            # no activation quantization of logits
            self.classifier.activation_quantizer = FP32Acts()
        elif quant_setup == "LSQ_paper":
            # Weights of the first layer
            self.conv_stem.activation_quantizer = FP32Acts()
            self.conv_stem.weight_quantizer.quantizer.n_bits = 8
            # Weights of the last layer
            self.classifier.activation_quantizer.quantizer.n_bits = 8
            self.classifier.weight_quantizer.quantizer.n_bits = 8
            # Set all QuantizedActivations to FP32
            for layer in self.blocks.modules():
                if isinstance(layer, QuantizedActivation):
                    layer.activation_quantizer = FP32Acts()
        elif quant_setup is not None and quant_setup != "all":
            raise ValueError(
                "Quantization setup '{}' not supported for EfficientNet lite".format(quant_setup)
            )

    def forward(self, x):
        # features
        x = self.conv_stem(x)
        x = self.blocks(x)
        x = self.conv_head(x)

        x = self.global_pool(x)
        x = x.flatten(1)
        return self.classifier(x)


def efficientnet_lite0_quantized(pretrained=True, model_dir=None, load_type="fp32", **qparams):
    if load_type == "fp32":
        # Load model from pretrained FP32 weights
        fp_model = create_model("efficientnet_lite0", pretrained=pretrained)
        quant_model = QuantizedEfficientNetLite(fp_model, **qparams)
    elif load_type == "quantized":
        # Load pretrained QuantizedModel
        print(f"Loading pretrained quantized model from {model_dir}")
        state_dict = torch.load(model_dir)
        fp_model = create_model("efficientnet_lite0")
        quant_model = QuantizedEfficientNetLite(fp_model, **qparams)
        quant_model.load_state_dict(state_dict)
    else:
        raise ValueError("wrong load_type specified")
    return quant_model
