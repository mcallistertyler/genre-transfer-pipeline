## Turns log magnitude spectrogram into pytorch tensor
from torchvision.transforms import ToTensor
from pathlib import Path
import argparse
import json
from PIL import Image
import torch
import os
from tqdm import tqdm

def inverse_normalisation(spec_names, norms, save_path, file_path):
    for idx, name in enumerate(spec_names):
        try:
            im = Image.open(file_path + name).convert('L')
            im = ToTensor()(im)
            name = name.replace('_fake.png', '')
            name = name.replace('_fake_B.png', '')
            name = name.replace('_fake_A.png', '')
            name = name.replace('.png', '')
            min_val = norms[name]['min']
            max_val = norms[name]['max']
            print('Denormalising', name)
            print('Values found are:\nmax:', max_val)
            print('min: ', min_val, '\n')
            denormalised_spec = im[0] * (max_val - min_val) + min_val
            torch.save(denormalised_spec, str(save_path)  + '/' + name + '-torch.pt')
        except FileNotFoundError:
            pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--input_file", type=str, default="input.txt")
    parser.add_argument("--manual-norm", type=bool, required=False)
    parser.add_argument("--file_path", type=str, required=False)
    parser.add_argument("--norm", type=str, default="normalisation_values.json")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    save_path = args.save_path
    args.save_path.mkdir(exist_ok=True, parents=True)
    file_name = args.input_file
    file_path = args.file_path
    norm_values = args.norm
    names = open(file_name, 'r')
    lines = names.readlines()
    lines = list(map(lambda s: s.strip(), lines)) # remove newline character
    lines = [track.replace('.wav', '') for track in lines] # remove .wav
    lines = [track.split("/")[-1] for track in lines]
    with open(norm_values) as json_file:
        norms = json.load(json_file)
        inverse_normalisation(lines, norms, save_path, file_path)

if __name__ == "__main__":
    main()
