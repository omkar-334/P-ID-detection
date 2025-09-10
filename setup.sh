#!/bin/bash
git clone https://github.com/omkar-334/P-ID-detection code
cd code || exit
pip install -r requirements.txt

# use hf token
huggingface-cli login

# to use hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

huggingface-cli download omkar334/PIDdataset --repo-type dataset annotations.zip --local-dir .
huggingface-cli download omkar334/PIDdataset --repo-type dataset images.zip --local-dir .

unzip annotations.zip -d dataset > /dev/null 2>&1
unzip images.zip -d dataset > /dev/null 2>&1
rm annotations.zip images.zip