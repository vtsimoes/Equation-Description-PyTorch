import pandas as pd
import os

args={
    'txt_dir': '../20_generate_textual_description/',             #directory for train images
    'output_dir': '../resize_training_equation_images/',     #diirectory for saving resized images
}

input_dir = args['txt_dir']

files = []
files_dir = []
for r,d,f in os.walk(input_dir):
    for file in f:
        if '.txt' in file:
            #images.append(os.path.join(r,file))
            files.append(file)
            files_dir.append(r)

image_file = []
captions = []
for image,dir in zip(files,files_dir):
        with open(os.path.join(dir,image), 'r+b') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(b'\t')
                image_file.append(line[0])
                captions.append(line[1].strip())
                #print(line)
image_file = [img_f.decode('utf-8') for img_f in image_file]
captions = [cap.decode('utf-8') for cap in captions]

df_captions = pd.DataFrame({'captions':captions,'image_file':image_file})
df_captions.to_json('../json_captions/json_captions.json')