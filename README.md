<img src="https://cdn-images-1.medium.com/max/1600/1*XsXMU-MwGH_O3Trt6OKhuw.png">

### [neural-style-tf](https://github.com/cysmith/neural-style-tf)

* **OS:** Ubuntu 18.04 LTS
* **CUDA:** 10
* **Python:** 2.7
* **Tensorflow-GPU:** 1.13
* **OpenCV:** 3.4.2
* **Scipy:** 1.1

#### Une image et un style
```
bash stylize_image.sh ./image_input/orange.jpg ./styles/cortes.jpg 
```
<img src="https://raw.githubusercontent.com/aquadzn/neural-style/master/examples/simple/init.png" height=32% width=32%> <img src="https://raw.githubusercontent.com/aquadzn/neural-style/master/examples/simple/style_0.png" height=32% width=32%> <img src="https://raw.githubusercontent.com/aquadzn/neural-style/master/examples/simple/result.png"  height=32% width=32%>
```
Single image elapsed time: 138.271806002
```

---

#### Une image, deux masques et deux styles
```
python neural_style.py 
--content_img face.jpg --style_mask --style_mask_imgs mask.png mask_inv.png --style_imgs picasso.jpg starry-night.jpg --style_imgs_weights 0.5 0.5
```
<img src="https://raw.githubusercontent.com/aquadzn/neural-style/master/examples/masks/mask.png" height=19% width=19%> <img src="https://raw.githubusercontent.com/aquadzn/neural-style/master/examples/masks/mask_inv.png" height=19% width=19%> <img src="https://raw.githubusercontent.com/aquadzn/neural-style/master/examples/masks/style_0.png"  height=19% width=19%> <img src="https://raw.githubusercontent.com/aquadzn/neural-style/master/examples/masks/style_1.png" height=19% width=19%> <img src="https://raw.githubusercontent.com/aquadzn/neural-style/master/examples/masks/result.png"  height=19% width=19%>
```
Single image elapsed time: 112.334724188
```

---

