<div style="text-align:center">
<img src="sth\SSTDP-LOGO.png" alt="SSTDP_Logo" width="900"/>
<h2>SSTDP: Supervised Spike Timing Dependent Plasticity for Efficient Spiking Neural Network Training</h2>
</div>
pytorch implementation of SSTDP for Efficient Spiking Neural Network Training

# Overview

## Description
- sstdp_module (`/code/sstdp_module.py`)
  - SpikeOnceNeuron: The neuron that will only spike once for each sample. Such neuron reduces the spike density while maintain spike information
  - stdp_update: The STDP update rule that direct the stdp update with gradient.
  - stdp_linear_container: The linear spiking neuron layer container.
  - StdpLinear: The callable linear model.
  - stdp_conv2d_container: The 2d convolution spiking neuron layer container.
  - StdpConv2d: The callable 2d convolution model.

- sstdp_train (`/code/sstdp_train.py`) run single, with parameters 
    ```bash
    python sstdp_train.py --threshold 100 --result_dir test_train/ --weight_decay 1e-5 --learning_rate 10
    ```
## Citation

We now have a [paper](#), titled "SSTDP: Supervised Spike Timing Dependent Plasticity for Efficient Spiking Neural Network Training", which is published in Frontiers in Neuroscience, Section Neuromorphic Engineering.
```bibtex
@article{liu2021sstdp,
  title={SSTDP: Supervised Spike Timing Dependent Plasticity for Efficient Spiking Neural Network Training},
  author={Liu, Fangxin and Zhao, Wenbo and Chen, Yongbiao and Wang, Zongwu and Yang, Tao and Li, JIANG},
  journal={Frontiers in neuroscience},
  year={2021},
  pages={1413},
  publisher={Frontiers}
}
```

## To-do

- [ ] ***Coming soon:*** Updated Code.
