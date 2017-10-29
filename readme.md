# Variational Autoencoder
[![dep1](https://img.shields.io/badge/Tensorflow-1.3+-blue.svg)](https://www.tensorflow.org/)

Implement variants of Variational Autoencoder by tensorflow.

### Models
- Conditional VAE
- Vanilla VAE
- CNN

## VAE


## CNN
At first, we have implemented CNN model for comparison.
The network consists of four CNN layer, and each layer includes max pooling and dropout.  
To run the test for mnist,

```
python train_cnn.py
```

For mnist classification, this model achieves over 98 % validation accuracy.

<p align="center">
  <img src="./img/cnn_log.png" width="1000">
  <br><i>learning log</i>
</p>

The description 