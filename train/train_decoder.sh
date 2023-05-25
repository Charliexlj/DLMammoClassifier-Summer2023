#!/bin/bash

wget -O mydataset.rar https://data.mendeley.com/public-files/datasets/ywsbh3ndr8/files/a49dc1cc-7a9e-422a-b3d4-a77cd908f594/file_downloaded
import rarfile

# Specify the path to the RAR file and the destination directory
rar_path = '/decoder/'
extract_dir = '/decoder/dataset'

os.makedirs(extract_dir, exist_ok=True)

# Open the RAR file and extract its contents to the destination directory
with rarfile.RarFile(rar_path) as rf:
  rf.extractall(extract_dir)