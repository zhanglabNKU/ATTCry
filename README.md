# ATTCry: Attention-based neural network model for protein crystallization prediction

Code for our paper "[ATTCry: Attention-based neural network model for protein crystallization prediction](https://www.sciencedirect.com/science/article/abs/pii/S0925231221012133)"

## Requirements

The code has been tested running under Python 3.6, with the following packages and their dependencies installed:
```
keras==2.2.0
numpy==1.16.0
tensorflow-gpu==1.9.0
pandas==0.20.3
scikit-learn==0.24.1
```

## Usage

```
git clone https://github.com/lengyuewuyazui/ATTCry.git
cd ATTCry
```

1. Run `dataprocess_data.py` to preprocess the data.
2. Run `train.py` to train the model.
3. Run `val.py` to evaluate the model.
4. Run `test.py` to test the model.

## Citation

```
@article{jin2021attcry,
  title={ATTCry: Attention-based neural network model for protein crystallization prediction},
  author={Jin, Chen and Gao, Jianzhao and Shi, Zhuangwei and Zhang, Han},
  journal={Neurocomputing},
  volume={463},
  pages={265-274},
  year={2021},
}
```
