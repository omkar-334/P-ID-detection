#!/bin/bash
git clone https://github.com/omkar-334/rcnn code
cd code || exit
pip install -r requirements.txt

# use hf token
huggingface-cli login

# to use hf_transfer for faster downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

huggingface-cli download omkar334/PIDdataset annotations.zip --repo-type dataset --local-dir .
huggingface-cli download omkar334/PIDdataset images.zip --repo-type dataset --local-dir .

unzip annotations.zip -d dataset > /dev/null 2>&1
unzip images.zip -d dataset > /dev/null 2>&1
rm annotations.zip images.zip
