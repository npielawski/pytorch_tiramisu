#!/bin/bash
# Run this script prior to running the tests

mkdir -p data

# Original image: https://www.pexels.com/photo/woman-in-pink-long-sleeve-shirt-carrying-white-puppy-5270669/
# Author: Julia Volk
wget --user-agent="Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1)" \
     --no-clobber \
     --output-document data/woman_in_pink.jpg \
     "https://images.pexels.com/photos/5270669/pexels-photo-5270669.jpeg?crop=entropy&cs=srgb&dl=pexels-julia-volk-5270669.jpg&fit=crop&fm=jpg&h=960&w=640"

# Checksums
sha256sum data/woman_in_pink.jpg -c checksum.sha256