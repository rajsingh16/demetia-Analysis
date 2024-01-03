import os
import pygit
import subprocess
import argparse

#path_toCheck = r"D:\\projects\\dmentia project\\opensmile-3.0-win-x64\\opensmile-3.0-win-x64\\config\\prosody"

#bin_dir = r"D:\\projects\\dmentia project\\opensmile-3.0-win-x64\\opensmile-3.0-win-x64\\bin"
git_cmd = "git clone https://github.com/puneetbawa/opensmile-2.3.0.git"

parser = argparse.ArgumentParser(description='Optional app description')

#Required positional argument

parser.add_argument('--wav_dir', type=str, help='A required data directory containing wav files')
argv = parser.parse_args()

#wav_dir = "D:\\projects\\dmentia project\\ADReSS-IS2020-train\\ADReSS-IS2020-data\\train\\Full_wave_enhanced_audio\\cc"
wav_dir = "D:\\projects\\dmentia project\\ADReSS-IS2020-train\\ADReSS-IS2020-data\\train\\Full_wave_enhanced_audio\\cd"
#wav_dir = wav_dir.strip()

#Path Validation of audio File

if(os.path.exists(wav_dir)):
    print("Path of all wav files for prosody extraction" + wav_dir)
else:
    print("Path doesn't exist. Check your audio file path and try again")
    exit(0)

if(os.path.exists(path_toCheck)):
    print("################################################################")
    print("############ STARTED EXTRACTION OF PROSODY FEATURES ############")
    print("################################################################")

    print("############CHECK LOG FILE smile_log_run.txt FOR ERRORS#########")
    os.chdir(wav_dir)
    print("Process changed its directory to " + wav_dir)
    files = os.listdir('.')
    os.mkdir("pros_feats1")

    #print("############ SUCCESSFULLY CREATED DIRECTORY PROSODY#############")

    for f_file in files:
        print(f_file)
        abs_path = os.path.abspath(f_file)
        sample_name = os.path.basename(f_file).split('.')[0]
        print(sample_name)
        args = path_toCheck + "\\SMILExtract -C " + path_toCheck + "\\config\\prosodyVir.conf -I " + abs_path + " -O " + "pros_feats\\" + sample_name + "_results.csv > smile_log_run.txt"
        print(args)
        subprocess.call(args, shell=True)
        print("######################################################################################")
        print("########### SUCCESSFULLY EXTRACTED PROSODY FEATURES FOR FILE" + f_file + "############")
        print("######################################################################################")

    print("################################################################")
    print("########### SUCCESSFULLY EXTRACTED PROSODY FEATURES ############")
    print("################################################################")
else:
    try:
        os.chdir(bin_dir)
        print("Changed Directory to " + bin_dir)
        print("cloning the smile Extract tool required for bin_dir")
        print(git_cmd)
        subprocess.call(git_cmd, shell=True)
        pygit.Repo.clone_from(git_URL, bin_dir)
        print("Process Sucessfully installed to" + bin_dir)
        print("Rerun - the same script")
    except:
        pass