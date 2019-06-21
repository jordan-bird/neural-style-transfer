# Neural style with Tensorflow-GPU 2.0.0-beta1

Neural style transfer algorithm implemented with last GPU version of Tensorflow ! <br>
@aquadzn (William Jacques)

## Installation (hardest part ( ͡ʘ ͜ʖ ͡ʘ) )

1) I advise you to follow the official Tensorflow installation tutorial to get latest version of tensorflow-gpu :
* PIP: https://www.tensorflow.org/install/pip
* Docker: https://www.tensorflow.org/install/docker

After installing Tensorflow, you can check that you have the correct version with ```import tensorflow as tf; tf.__version__``` and get ```2.0.0-beta1```

2) Then you can clone my repository ```git clone https://github.com/aquadzn/neural-style-transfer.git```

3) Install imageio package by running ```pip install imageio```

PS : If you're having trouble with installing Tensorflow or running it with your GPU, I recommend you to correctly install CUDA and CUDNn libraries by following [Tensorflow's official tutorial](https://www.tensorflow.org/install/gpu#ubuntu_1804_cuda_10)

## Usage

```
usage: model.py [-h] [--output OUTPUT] [--learning_rate LEARNING_RATE]
                [--size SIZE]
                imagelink stylelink

positional arguments:
  imagelink             URL of the input image (optional, example:
                        https://website.com/image.jpg)
  stylelink             URL of the style image to apply (optional, example:
                        https://website.com/style.jpg)

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT       File name of the output image (default: result.jpg)
  --learning_rate LEARNING_RATE
                        Choose a learning rate (default: 0.02)
  --size SIZE           Modify the size of the output image (default: 512)
```
### Basic usage
```python model.py imagelink stylelink```

If you are using Windows, you will need to delete the two .jpg images in the keras/datasets/ folder in order to try again with others images.
On Linux systems, it will automatically remove them if the .keras/ folder is where the command 'cd' returns.

#### Example:
```
python model.py https://acadienouvelle-6143.kxcdn.com/wp-content/uploads/2019/03/lion-3049884_960_720.jpg https://definicion.mx/wp-content/uploads/literatura/Expresionismo.jpg
```
![image](https://i.imgur.com/0NPjxRo.jpg "Exemple")
---
### With optional arguments
```python model.py imagelink stylelink --learning_rate --size```

#### Example:
```
python model.py https://acadienouvelle-6143.kxcdn.com/wp-content/uploads/2019/03/lion-3049884_960_720.jpg https://definicion.mx/wp-content/uploads/literatura/Expresionismo.jpg --learning_rate 0.2 --size 1024
```

## Setup

* **OS:** Ubuntu 18.04 LTS
* **CUDA:** 10
* **Python:** 3.7
* **Tensorflow-GPU:** 2.0.0-beta1
