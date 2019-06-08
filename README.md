### Neural style transfer
* **OS:** Ubuntu 18.04 LTS
* **CUDA:** 10
* **Python:** 2.7
* **Tensorflow-GPU:** 1.13
* **OpenCV:** 3.4.2
* **Scipy:** 1.1


#### Une image et un style
```
bash stylize_image.sh ./image_input/taj_mahal.jpg ./styles/cortes.jpg 
```

---

#### Une image, deux masques et un style
```
python neural_style.py
--content_img "face.jpg" --content_img_dir "image_input" --style_mask --style_mask_imgs "mask.png" --style_imgs "nuit.jpg" --style_imgs_dir "styles"
```

---

#### Une image, deux masques et deux styles
```
python neural_style.py 
--content_img "face2.png" --content_img_dir "image_input"--style_mask --style_mask_imgs "mask2.png" --style_imgs "nuit.jpg" --style_imgs_dir "styles"
```


```
Single image elapsed time: 112.334724188
```
