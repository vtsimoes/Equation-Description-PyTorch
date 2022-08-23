from data_loader import get_loader
import torch
import torchvision.transforms as transforms
import pickle
from build_vocab import Vocabulary

args={
    'caption_path':'../json_captions/json_captions.json',                #path for train annotation file
    'vocab_path':'../data/vocab.pkl',
    'image_dir': '../resize_training_equation_images',                                          #path for saving vocabulary wrapper
    'batch_size': 10,
    'num_workers': 1,
    'crop_size': 224                                                                                          #minimum word count threshold
}


# Load vocabulary wrapper.
with open(args['vocab_path'], 'rb') as f:
    vocab = pickle.load(f)

f.close()

# Image preprocessing
transform = transforms.Compose([ 
    transforms.RandomCrop(args['crop_size']),
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(), 
    transforms.Normalize((0.9638, 0.9638, 0.9638), 
                            (0.1861, 0.1861, 0.1861))])

# Build data loader
data_loader = get_loader(args['image_dir'], args['caption_path'], vocab, 
                             transform, args['batch_size'],
                             shuffle=True, num_workers=args['num_workers'])

#for test_images, test_labels in data_loader:  
    #sample_image = test_images[0]    # Reshape them according to your needs.
    #sample_label = test_labels[0]
for batch in data_loader:
    print(batch["x"].shape, batch["y"])