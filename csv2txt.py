import csv
import os

import argparse

cwd = os.getcwd()

# print(cwd)

parser = argparse.ArgumentParser(description='Optional app description')

# Required possible positional arguement

parser.add_argument('--csv_dir', type=str,
                    help='A required data directory containing wav files'
                    )
argv = parser.parse_args()

csv_dir = argv.csv_dir

# Path Validation for the existance of the csv file

if os.path.exists(csv_dir):
    print('Path of all csv files for formating csv is' + csv_dir)
    txt_dir = csv_dir + '/txt_pro_feats'
    print('Creating New DIrectory' + txt_dir)
    os.mkdir(txt_dir)
else:
    print("Path doesn't exist. Check for the path of csv files and try again")
    exit(0)

for file in os.listdir(csv_dir):  # use the directory name here

    (file_name, file_ext) = os.path.splitext(file)
    if file_ext == '.csv':
        csv_f = csv_dir + '/' + file_name +'.csv'
        with open(csv_f, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # skips the header row of the csv file
            newfile = txt_dir +'/' + file_name + '_npro.txt'

            for line in csv_reader:
                with open(newfile, 'a') as new_txt:  # new file has .txt extn
                    txt_writer = csv.writer(new_txt, delimiter=',')  # writefile
                    txt_writer.writerow(line)  # write the lines to file`