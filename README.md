# Overcoming Oscillations in Quantization-Aware Training
This repository containes the implementation and experiments for the paper presented in

**Markus Nagel<sup>\*1</sup>, Marios Fournarakis<sup>\*1</sup>,  Yelysei Bondarenko<sup>1</sup>, 
Tijmen Blankevoort<sup>1</sup> "Overcoming Oscillations in Quantization-Aware Training", ICML 
2022.** [[ArXiv]](https://arxiv.org/abs/2203.11086)

*Equal contribution
<sup>1</sup> Qualcomm AI Research (Qualcomm AI Research is an initiative of Qualcomm Technologies, Inc.)

You can use this code to recreate the results in the paper.
## Reference
If you find our work useful, please cite
```
@InProceedings{pmlr-v162-nagel22a,
  title = 	 {Overcoming Oscillations in Quantization-Aware Training},
  author =       {Nagel, Markus and Fournarakis, Marios and Bondarenko, Yelysei and Blankevoort, Tijmen},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {16318--16330},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/nagel22a/nagel22a.pdf},
  url = 	 {https://proceedings.mlr.press/v162/nagel22a.html}
  }
```

## Method and Results

When training neural networks with simulated quantization, we observe that quantized weights can,
rather unexpectedly, oscillate between two grid-points. This is an inherent issue problem caused 
by the straight-through-estimator (STE). In our paper, we delve deeper in this little understood 
phenomenon and show that oscillations harm accuracy by corrupting the EMA statistics of the 
batch-normalization layers and by preventing convergence to local mimima. 

<p align="center">
    <img src="display/toy_regression.gif " width="425"/>
</p>

We propose two novel methods to tackle oscillations at their source: **oscillations dampening** 
and **iterative state freezing** We demonstrate  that our algorithms achieve state-of-the-art 
accuracy for low-bit (3 & 4 bits) weight and activation quantization of efficient architectures, 
such as MobileNetV2, MobileNetV3, and EfficentNet-lite on ImageNet.


## How to install
Make sure to have Python ≥3.6 (tested with Python 3.6.8) and 
ensure the latest version of `pip` (**tested** with 21.3.1):
```bash
source env/bin/activate
pip install --upgrade --no-deps pip
```

Next, install PyTorch 1.9.1 with the appropriate CUDA version (tested with CUDA 10.0, CuDNN 7.6.3):
```bash
pip install torch==1.9.1 torchvision==0.10.1
```

Finally, install the remaining dependencies using pip:
```bash
pip install -r requirements.txt
```

## Running experiments
The main run file to reproduce all experiments is `main.py`. 
It contains commands for quantization-aware training (QAT) and validating quantized models.
You can see the full list of options for each command using `python main.py [COMMAND] --help`.
```bash
Usage: main.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  train-quantized
```

## Quantization-Aware Training (QAT)
All models are fine-tuned starting from pre-trained FP32 weights. Pretrained weights may be found here

- [MobileNetV2](https://drive.google.com/open?id=1jlto6HRVD3ipNkAl1lNhDbkBp7HylaqR)
- EfficientNet-Lite: pretrained weights from [repository](https://github.com/rwightman/pytorch-image-models/) (downloaded at runtime)

## MobileNetV2

To train with **oscillations dampening** run:
```bash
python main.py train-quantized  --arhcitecture mobilenet_v2_quantized
--images-dir path/to/raw_imagenet --act-quant-method MSE  --weight-quant-method MSE 
--optimizer SGD --weight-decay 2.5e-05 --sep-quant-optimizer 
--quant-optimizer Adam --quant-learning-rate 1e-5 --quant-weight-decay 0.0 
--model-dir /path/to/mobilenet_v2.pth.tar --learning-rate-schedule cosine:0
# Dampening loss configurations 
--oscillations-dampen-weight 0 --oscillations-dampen-weight-final 0.1 
# 4-bit best learning rate
--n-bits 4 --learning-rate 0.0033 
# 3-bits best learning rate
--n-bits 3 --learning-rate 0.01
```

To train with **iterative weight freezing** run:
```bash
python main.py train-quantized  --arhcitecture mobilenet_v2_quantized
--images-dir path/to/raw_imagenet --act-quant-method MSE  --weight-quant-method MSE 
--optimizer SGD  --sep-quant-optimizer 
--quant-optimizer Adam --quant-learning-rate 1e-5 --quant-weight-decay 0.0 
--model-dir /path/to/mobilenet_v2.pth.tar --learning-rate-schedule cosine:0
# Iterative weight freezing configuration
--oscillations-freeze-threshold 0.1
# 4-bit best configuration
--n-bits 4 --learning-rate 0.0033 --weight-decay 5e-05 --oscillations-freeze-threshold-final 0.01 
# 3-bit best configuration
--n-bits 3 --learning-rate 0.01 --weight-decay 2.5e-05 --oscillations-freeze-threshold-final 0.011
```

For end user's convenience, bash scripts are provided under `/bash/` for reproducing our experiments.
```bash
./bash/train_mobilenetv2.sh --IMAGES_DIR path_to_raw_imagenet --MODEL_DIR path_to_pretrained_weights # QAT training of MobileNetV2 with defaults (method 'freeze' and 3 bits)
./bash/train_efficientnet.sh --IMAGES_DIR path_to_raw_imagenet --METHOD damp --N_BITS 4
```
