# MIC-RWKV
## Data Preprocessing
- Download data by running download_data.py<br>
- Enter the raw_data directory, and get AMRs with RGI: for file in *.fasta; do rgi main --input_sequence "${file}" --output_file data/amr_raw_data/"${file}" --clean; done
## Training
- Train the model with: python MIC-RWKV/train.py
