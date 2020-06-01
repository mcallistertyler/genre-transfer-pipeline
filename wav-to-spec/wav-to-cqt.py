## Turns wav files into CQT Spectrogram
#from mel2wav.dataset import AudioDataset
from dataset import AudioConversionDataset
import os
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import argparse
import json
from nnAudio import Spectrogram
import torch
from tqdm import tqdm

normalisation_dict = {}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--save_path", type=str, default="./")
    parser.add_argument("--save_type", type=str, default="png")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    file_name = args.input_file
    save_path = args.save_path
    spec_layer = Spectrogram.CQT1992v2(sr=22050, n_bins=84, hop_length=256, pad_mode='constant', device='cuda:0', verbose=False, trainable=False, output_format='Magnitude')
    transformedSet = AudioConversionDataset(file_name, 22050 * 4, sampling_rate=22050, augment=False)
    transformedLoader = DataLoader(transformedSet, batch_size=1)
    f = open(file_name, 'r')
    lines = f.readlines()
    lines = list(map(lambda s: s.strip(), lines)) #remove newline character
    lines = [track.replace('.wav', '') for track in lines] #remove .wav
    lines = [track.split("/")[-1] for track in lines]
    if len(lines) != len(transformedLoader):
        print('Differences in wavs found and whats in input_audio.txt')
        return

    for i, x in tqdm(enumerate(transformedLoader), ascii=True, desc='Making spectrogram representations'):
        x_t = x[0]
        fname = os.path.basename(x[1][0]).replace('.wav', '')
        x_t = x_t.cuda()
        s_t = spec_layer(x_t).detach()
        s_t = torch.log(torch.clamp(s_t, min=1e-5))
        if args.save_type == 'pt':
            torch.save(s_t.cuda(), save_path + fname + '.pt')
        else:
            save_image(s_t.cuda(), save_path + fname + '.png', normalize=True)
            min_value = torch.min(s_t.cuda()).item()
            max_value = torch.max(s_t.cuda()).item()
            normalisation_dict[fname] = { "min": min_value, "max": max_value }
    with open(save_path + 'normalisation_values.json', 'w') as outfile:
        json.dump(normalisation_dict, outfile, indent=4)

if __name__ == "__main__":
    main()
