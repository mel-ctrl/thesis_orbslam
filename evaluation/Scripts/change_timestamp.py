import argparse
import sys
import os
import numpy


def read_file_list(filename):
    file = open(filename)
    save_path = (filename.split(".")[0]) + ".txt"
    lines = file.readlines()
    with open(save_path, "a") as new_file:
        for line in lines:
            new_line = ('{:f}'.format(float(line.split(" ")[0])*10**9) + " " + " ".join(line.split(" ")[1: None]))
            new_file.write(new_line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    This script takes data files with timestamps''')
    parser.add_argument('file', help='first text file (format: timestamp data)')
    args = parser.parse_args()
    #args = parser.parse_args(["/home/meltem/Downloads/mc0003_20220603_145039.gps"])
    read_file_list(args.file)