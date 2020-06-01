#!/bin/bash

echo "Sourcing GPU."
source ./set_env.sh 0
echo "Choose the genre of your music track(s)"
echo "Vaporwave, Electronic, Pop, Instrumental, Hip-Hop"

arr=()
arr+=('vaporwave')
arr+=('electronic')
arr+=('pop')
arr+=('instrumental')
arr+=('hip-hop')
genre_origin=$1
genre_origin="$(echo $genre_origin | tr '[:upper:]' '[:lower:]')"

if [[ " ${arr[*]} " == *"$genre_origin"* ]];
then
    echo "Genre chosen."
else
    echo "Not a valid genre."
    exit 1
fi

echo "Choose the genre to transfer your music track to:"
echo "Vaporwave, Electronic, Pop, Instrumental, Hip-Hop"
#read genre_transfer
genre_transfer=$2
genre_transfer="$(echo $genre_transfer | tr '[:upper:]' '[:lower:]')"

if [[ " ${arr[*]} " == *"$genre_transfer"* ]];
then
    echo "Genre chosen."
else
    echo "Not a valid genre."
    exit 1
fi

if [[ $genre_origin = $genre_transfer ]];
then
    echo "Can't transfer to the same genre."
    exit 1
fi

rm input.txt
rm concat.txt
rm -r cqt-dump
rm -r output
rm to-tensor.txt
mkdir output
mkdir cqt-dump
readlink -f music/splits/*.wav >> input.txt
python wav-to-spec/wav-to-cqt.py --input_file input.txt --save_path cqt-dump/
mkdir cqt-dump/pngs/
mv cqt-dump/*.png cqt-dump/pngs
num_specs=$(ls cqt-dump/pngs | wc -l)
transfer_model="${genre_origin}-and-${genre_transfer}"
echo $num_specs
echo $transfer_model
rm -r cqt-dump/${transfer_model}
python style-transfer/cyclegan-final/test.py --dataroot cqt-dump/pngs --name $transfer_model --model test --no_dropout --num_test $num_specs --direction AtoB --results_dir  cqt-dump/ --checkpoints_dir ./style-transfer/cyclegan-final/checkpoints --preprocess none
rm cqt-dump/${transfer_model}/test_latest/images/*_real.png
rm input.txt
readlink -f cqt-dump/${transfer_model}/test_latest/images/*_fake.png >> to-tensor.txt 
rm -r spec-reconstruction/cqtgan-final/scripts/tensors
mkdir spec-reconstruction/cqtgan-final/scripts/tensors
python wav-to-spec/spec-to-tensor.py --input_file to-tensor.txt --save_path spec-reconstruction/cqtgan-final/scripts/tensors --norm cqt-dump/normalisation_values.json --file_path cqt-dump/${transfer_model}/test_latest/images/
source spec-reconstruction/cqtgan-final/set_env.sh 0
python spec-reconstruction/cqtgan-final/scripts/generate_from_spectrogram.py --pt_path spec-reconstruction/cqtgan-final/scripts/tensors/ --load_path spec-reconstruction/cqtgan-final/${genre_transfer}/baseline --save_path ./output
for f in output/*.wav; do echo "file '$f'" >> concat.txt; done
