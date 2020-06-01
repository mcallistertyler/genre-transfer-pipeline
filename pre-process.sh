#Split up file
echo "Enter file to be pre-processed. This will delete currently pre-processed files."
rm -r music/splits/*
read file_name
ffmpeg -i "music/"$file_name -c copy -map 0 -segment_time 00:00:05 -f segment -reset_timestamps 1 "music/splits/"output%03d.wav
for file_splits in "music/splits/"*.wav; do
    timecheck=`ffprobe -i ${file_splits} -show_entries format=duration -v quiet -of csv="p=0"`
    float_time=$(echo "scale=3; $timecheck" | bc)
    if (( $(echo "$float_time < 4.0" | bc -l) ));
    then
        rm $file_splits
    fi
done
