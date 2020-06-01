import os

from mel2wav import MelVocoder

from pathlib import Path
from tqdm import tqdm
import argparse
import librosa
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--pt_path", type=Path, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    vocoder = MelVocoder(args.load_path)

    args.save_path.mkdir(exist_ok=True, parents=True)
    
    for i, fname in tqdm(enumerate(args.pt_path.glob('*.pt'))):
        wavname = os.path.splitext(fname.name)[0] + '.wav'
        print('fname', fname)
        print('wavname', wavname)
        spectrogram = torch.load(fname)
        if (len(spectrogram.shape) == 2):
            spectrogram = spectrogram.unsqueeze(0)
        reconstruction = vocoder.inverse(spectrogram).squeeze().cpu().numpy()
        librosa.output.write_wav(str(args.save_path) + '/' + wavname, reconstruction, sr=22050)

if __name__ == "__main__":
    main()
