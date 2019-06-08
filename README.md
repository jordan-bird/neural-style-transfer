### Neural style transfer
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

```
Single image elapsed time: 138.271806002
```

---

#### Une image, deux masques et deux styles
```
python neural_style.py 
--content_img face.jpg --style_mask --style_mask_imgs mask.png mask_inv.png --style_imgs picasso.jpg starry-night.jpg --style_imgs_weights 0.5 0.5
```

```
Single image elapsed time: 112.334724188
```

---