# Auteur : William Jacques (https://github.com/aquadzn/)

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import imageio
import time
import argparse


desc = "Neural Style Transfer implementation with Tensorflow 2 (tensorflow-gpu==2.0.0-beta1)" + "\n" + "You need to pass as argument at least 'imagelink' and 'stylelink'. You can also change the learning rate or the size if you want. The bigger the output image size is, the longer it will take to run. Default output size takes about 35 seconds."
parser = argparse.ArgumentParser(description=desc)

parser.add_argument('imagelink',
                    type=str,
                    help="URL of the input image (optional, example: https://website.com/image.jpg)",
                    )

parser.add_argument('stylelink',
                    type=str,
                    help="URL of the style image to apply (optional, example: https://website.com/style.jpg)",
                    )

parser.add_argument('--output',
                    type=str,
                    help="File name of the output image (default: result.jpg)",
                    default='result.jpg')

parser.add_argument('--learning_rate',
                    type=float,
                    help="Choose a learning rate (default: 0.02)",
                    default=0.02,
                    )

parser.add_argument('--size',
                    type=int,
                    help="Modify the size of the output image (default: 512)",
                    default=512,
                    )

args = parser.parse_args()


if (args.imagelink is not None and args.stylelink is not None):
    print('')
    original_path = tf.keras.utils.get_file("image.jpg" ,args.imagelink)
    style_path = tf.keras.utils.get_file("style.jpg", args.stylelink)
    print("\n" + "╔ ——————————————————————————— ╗")
    print('  Images successfully loaded.')
    print("╚ ——————————————————————————— ╝" + "\n")
else:
    print("Please select a valid URL or try to upload it on hosting website.")

def load_img(img_path):
    max_dim = args.size
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

original_img = load_img(original_path)
style_img= load_img(style_path)

x = tf.keras.applications.vgg19.preprocess_input(original_img*255)
x = tf.image.resize(x, (224, 224))
vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
r = vgg(x)
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

original_layers = ['block5_conv2']

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_original_layers = len(original_layers)
num_style_layers = len(style_layers)


def vgg_layers(layer_names):

    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_img*255)


def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)


class StyleOriginalModel(tf.keras.models.Model):
    def __init__(self, style_layers, original_layers):
        super(StyleOriginalModel, self).__init__()
        self.vgg = vgg_layers(style_layers + original_layers)
        self.style_layers = style_layers
        self.original_layers = original_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, original_outputs = (outputs[:self.num_style_layers],
                                           outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        original_dict = {original_name: value
                         for original_name, value
                         in zip(self.original_layers, original_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'original': original_dict, 'style': style_dict}

extractor = StyleOriginalModel(style_layers, original_layers)

results = extractor(tf.constant(original_img))
style_results = results['style']

style_targets = extractor(style_img)['style']
original_targets = extractor(original_img)['original']

image = tf.Variable(original_img)

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


opt = tf.optimizers.Adam(learning_rate=args.learning_rate, beta_1=0.99, epsilon=1e-1)

original_weight = 1e4
style_weight = 1e-2

def style_original_loss(outputs):
    style_outputs = outputs['style']
    original_outputs = outputs['original']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    original_loss = tf.add_n([tf.reduce_mean((original_outputs[name]-original_targets[name])**2)
                             for name in original_outputs.keys()])
    original_loss *= original_weight / num_original_layers
    loss = style_loss + original_loss
    return loss


def high_pass_x_y(image):
  x_var = image[:,:,1:,:] - image[:,:,:-1,:]
  y_var = image[:,1:,:,:] - image[:,:-1,:,:]

  return x_var, y_var

x_deltas, y_deltas = high_pass_x_y(original_img)
x_deltas, y_deltas = high_pass_x_y(image)
sobel = tf.image.sobel_edges(original_img)

def total_variation_loss(image):
  x_deltas, y_deltas = high_pass_x_y(image)
  return tf.reduce_mean(x_deltas**2) + tf.reduce_mean(y_deltas**2)

total_variation_weight = 1e8

@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_original_loss(outputs)
    loss += total_variation_weight*total_variation_loss(image)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))


image = tf.Variable(original_img)


start = time.time()

epochs = 10
steps = 100

step = 0
for n in range(epochs):
  for m in range(steps):
    step += 1
    train_step(image)

end = time.time()
print('\n')
print("╔ ———————————————————————— ╗")
print("  Total time: {:.1f} seconds".format(end-start))
print("╚ ———————————————————————— ╝" + "\n")

output_name = args.output

if not os.path.exists('images'):
    os.mkdir('images')
    imageio.imwrite(f"images/{output_name}", image[0])
    print('\n')
    print("╔ ————————————————————————————— ╗")
    print(f"  {output_name} correctly created.")
    print("╚ ————————————————————————————— ╝")
    print('\n')
else:    
    imageio.imwrite(f"images/{output_name}", image[0])
    print('\n')
    print("╔ ————————————————————————————— ╗")
    print(f"  {output_name} correctly created.")
    print("╚ ————————————————————————————— ╝")
    print('\n')

if (args.imagelink is not None and args.stylelink is not None):
    if os.name == 'posix':
        os.system('cd  && rm .keras/datasets/image.jpg .keras/datasets/style.jpg && cd -')
        print("╔ ——————————————————————————————————————————————————————————————————— ╗")
        print("  URL images successfully deleted. You can try again with others URLs.")
        print("╚ ——————————————————————————————————————————————————————————————————— ╝" + "\n")
    else:
        print("If you want to retry with others images links, please remove 'image.jpg' and 'style.jpg' stored in your '.keras/datasets/' directory." + "\n")
else: exit


def main():
  global args
  
if __name__ == '__main__':
  main()