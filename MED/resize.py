import os
from PIL import Image
import os

args={
    'image_dir': '../4_generate_crop_image/',             #directory for train images
    'output_dir': '../resize_training_equation_images/',     #diirectory for saving resized images
    'image_size': 224                                                # 256 size for image after processing for training images
}

def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.ANTIALIAS)

def resize_images(image_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #images = os.listdir(image_dir)
    images = []
    images_dir = []
    for r,d,f in os.walk(image_dir):
        for file in f:
            if '.png' in file:
                #images.append(os.path.join(r,file))
                images.append(file)
                images_dir.append(r)

    for im in images[0:10]:
        print(im)

    num_images = len(images)
    #for i, image in enumerate(images):
    i = 0
    for image,dir in zip(images,images_dir):
        with open(os.path.join(dir,image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                img.save(os.path.join(output_dir, image), img.format)
        if i % 1000 == 0:
            print ("[%d/%d] Resized the images and saved into '%s'."
                   %(i, num_images, output_dir))
        i=i+1

def main():
    image_dir = args['image_dir']
    output_dir = args['output_dir']
    image_size = [args['image_size'], args['image_size']]
    resize_images(image_dir, output_dir, image_size)


if __name__ == '__main__':
    main()
