#!/bin/bash


config_file="D:\projects\dmentia project\opensmile-3.0.0\config\is09-13/IS13_ComParE.conf"
audio_dir="D:\projects\dmentia project\ADReSS-IS2020-train\ADReSS-IS2020-data\train\Normalised_audio-chunks\cc"

for entry in "$audio_dir"/*.wav; do
  fn="${entry%.*}"

  output_file="compare_${fn##*/}.csv"

  SMILExtract -C "$config_file" -I "$entry" -csvoutput "$output_file"
done