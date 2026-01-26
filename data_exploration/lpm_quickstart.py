import os
import sys
import math
import random
import glob
import pdb
import scipy
import matplotlib.pyplot as plt
import numpy as np
import json as json
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from transformers import AutoTokenizer, BertModel, AutoModelForQuestionAnswering, BertTokenizerFast, ViltProcessor, ViltModel

from PIL import Image


class LPDataset(data.Dataset):

  def __init__(self, cap_json, fig_json, connect_json, rootdir, wemb_type, transform=None, ids=None, skip_images=False):
    """
    Args:
      json: full_dataset.
      transform: transformer for image.
      skip_images: if True, skip loading actual image files
    """

    # if ids provided by get_paths, use split-specific ids
    self.ids = [item for sublist in list(connect_json.values()) for item in sublist] # all ids
    self.transform = transform
    self.cap_json = cap_json
    self.fig_json = fig_json
    self.connect_json = connect_json
    self.root = rootdir
    self.skip_images = skip_images

    self.wemb_type = wemb_type

    if 'bert' in self.wemb_type:
      self.bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    if 'vilt' in self.wemb_type:
      self.vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

  def __len__(self):
    return len(self.ids)


  def __getitem__(self, index):
    sent, img_id, path, image, ocr, pointers = self.get_raw_item(index)
    ocr_text = " ".join([o['text'] for o in ocr])

    if image is not None and self.transform is not None:
      image = self.transform(image)
    elif image is None:
      # Create dummy image tensor if images are skipped
      image = torch.zeros(3, 224, 224)

    if 'bert' in self.wemb_type:
      spoken_text = self.bert_tokenizer(sent,max_length=512, padding='max_length',truncation=True, return_tensors='pt')
      spoken_target = spoken_text['input_ids'].squeeze()
      ocr_target = self.bert_tokenizer(ocr_text,max_length=40, padding='max_length',truncation=True, add_special_tokens = False, return_tensors='pt')['input_ids']

    if 'vilt' in self.wemb_type:
      if image is not None:
        fig_ocr = self.vilt_processor(image, ocr_text, max_length=40, padding='max_length',truncation=True, return_tensors="pt")
      else:
        # Skip vilt processing if no image
        fig_ocr = None

      pointer_target =  torch.zeros_like(spoken_target)
      for point in pointers:
        inds = torch.Tensor([i for i, x in enumerate(spoken_text.word_ids()) if x == point]).long()
        pointer_target[inds] = 1
      return fig_ocr, spoken_target, pointer_target, index, img_id

    return  image, spoken_target, ocr_target, index, img_id

  def get_raw_item(self, index):

    img_id = self.ids[index]

    words = []
    captions = self.fig_json[img_id]['captions']
    for word in captions:
      words.append(str(word['Word']))

    sentence = " ".join(words)


    scene_id = self.fig_json[img_id]['scene_id']

    if scene_id[0] == os.path.basename(self.root):
      _ = scene_id.pop(0)

    # Correctly construct the path by joining scene_id elements
    path = os.path.join(self.root, *scene_id)+".jpg"
    
    # Skip image loading if flag is set
    if self.skip_images:
      image = None
      ocr = self.fig_json[img_id]['slide_text']
      pointers = self.fig_json[img_id]['pointers']
      return sentence, img_id, path, image, ocr, pointers
    
    # Original image loading code
    image = Image.open(path).convert('RGB')

    #crop image here
    cr = self.fig_json[img_id]['slide_figure']
    x = cr["left"]
    y = cr["top"]
    w = cr["width"]
    h = cr["height"]

    # print(cr)
    image = image.crop((x,y,x + w, y + h))
    ocr =  self.fig_json[img_id]['slide_text']
    pointers = self.fig_json[img_id]['pointers']

    return sentence, img_id, path, image, ocr, pointers


def get_image_transform():
  normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
  t_list = []

  t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
  t_end = [transforms.ToTensor(), normalizer]
  transform = transforms.Compose(t_list + t_end)
  return transform

def get_image_transform_default():
  t_list = []

  t_list = [transforms.Resize(256)]
  t_end = [transforms.ToTensor()]
  transform = transforms.Compose(t_list + t_end)
  return transform

def default_collate_fn(data, caption_lim = 512):
  # Sort a data list by sentence length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, spoken_target, ocr_target, index, img_ids = zip(*data)
    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    cap_lengths = torch.tensor([len(cap) if len(cap) <= caption_lim else caption_lim for cap in spoken_target])
    spoken_output = torch.zeros(len(spoken_target), caption_lim).long()


    for i, cap in enumerate(spoken_target):
      end = cap_lengths[i]
      if end <= caption_lim:
        spoken_output[i, :end] = cap[:end]
      else:
        cap_lengths[i] = caption_lim
        spoken_output[i, :end] = cap[:caption_lim]
    return images, spoken_output, ocr_target, cap_lengths, index, img_ids

def main():
    #define which speaker you want to load - this is the only thing you want to change
    sp = 'anat-1'

    #define where data is uploaded, by default, the following should work
    root_data_dir = 'lpm_data'
    sp_data_dir = '{}/{}'.format(root_data_dir, sp)

    #read jsons
    with open("{}/{}/{}_figs.json".format(root_data_dir, sp,sp), 'r') as f:
        fig_json = json.loads(f.read())

    with open("{}/{}/{}.json".format(root_data_dir, sp,sp), 'r') as j:
        cap_json = json.loads(j.read())

    with open("{}/{}/{}_capfig.json".format(root_data_dir, sp,sp), 'r') as c:
        connect_json = json.loads(c.read())

    #load data with skip_images=True
    transform = get_image_transform_default() #use get_image_transform() for normalization + cropping + resizing
    wemb_type = 'bert'
    dataset = LPDataset(cap_json, fig_json, connect_json, sp_data_dir, wemb_type, transform, skip_images=True)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=1,
                                            shuffle=True,
                                            pin_memory=False,  # Changed to False for MPS compatibility
                                            num_workers=0,  # Changed to 0 to avoid multiprocessing issues
                                            collate_fn = default_collate_fn
                                            )
    
    dataiter = iter(loader)
    images, spoken_output, ocr_target, cap_lengths, index, img_ids  = next(dataiter)
    bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    decoded_text = bert_tokenizer.decode(spoken_output[0], skip_special_tokens=True)
    
    print("Decoded spoken text:")
    print(decoded_text)
    print("\nOCR target shape:", ocr_target[0].shape)
    print("Caption length:", cap_lengths[0].item())
    print("Image ID:", img_ids[0])

if __name__ == "__main__":
    main()