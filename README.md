# Characterizing the Representation Disparity of Differential Privacy

This repo contains the main source code for the experiments and analysis in the paper "Characterizing the Representation Disparity of Differential Privacy". It also links to repositories for some of the supplementary experiments in the paper, which are contained in separate repos.

# Usage

Configure environment by running: `pip3 install -r REQUIREMENTS.txt`.

For our experiments, we use Python 3.6.7 and a single NVIDIA RTX 2080 Ti GPU with 11GB of RAM.

To run a training experiment, use `train.py` and the `--params` flag to provide a path to a `.yaml` file containing the experiment parameters. For example:

``` 
python3 train.py --params params/params_celeba_nodp.yaml
```

# Datasets

This repository uses the following datasets:

* MNIST (included in `torchvision.datasets`)
* CIFAR-10 (included in `torchvision.datasets`)
* CelebA ([link](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)) 
* Labeled Faces in the Wild (LFW) ([link](http://vis-www.cs.umass.edu/lfw/)) 
* German Credit ([link](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)))  
* COMPAS ([link](https://github.com/propublica/compas-analysis))
* Adult Income ([link](https://archive.ics.uci.edu/ml/datasets/adult))

## MMNIST and MC10 datasets

We also generate the MMNIST and MC10 datasets using the original MNIST and CIFAR-10 datasets. Details of the dataset generation process are described in the paper. The classes implementing these datasets are in `utils/mmnist_dataset.py` and `utils/mc10_dataset.py`, respectively. Samples from the datasets are shown below.

![MMNIST and MC10 sample](/img/grouped-datasets-both-example.png)

We gratefully acknowledge the authors of the paper "Differential Privacy Has Disparate Impact on Model Accuracy" [arXiv](https://arxiv.org/abs/1905.12101) for providing their open-source [code](https://github.com/ebagdasa/differential-privacy-vs-fairness), which was used to build this repository.

As in [4], use `compute_dp_sgd_privacy.py` copied from public [repo](https://github.com/tensorflow/privacy).

Except where otherwise indicated, we use the implementation of DPSGD from [4], which is based on TF Privacy [repo](https://github.com/tensorflow/privacy) and [1], [2], and [3] below.

[1] M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, and L. Zhang. Deep learning with differential privacy. In CCS, 2016.

[2] H. B. McMahan and G. Andrew. A general approach to adding differential privacy to iterative training procedures. arXiv:1812.06210, 2018

[3] H. B. McMahan, D. Ramage, K. Talwar, and L. Zhang. Learning differentially private recurrent language models. In ICLR, 2018

[4] E. Bagdasaryan, V. Shmatikov. Differential Privacy Has Disparate Impact on Model Accuracy. In NeurIPS, 2019.
