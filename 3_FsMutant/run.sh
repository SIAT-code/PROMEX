# 1. processing data
### 1) Process the wild-type sequences and the DataFrame
python preprocess.py
### 2) Vectorize the wild-type sequences
python retrieve.py -m vectorize -md esm2 -b 8
### 3) Compute an association matrix using the vectorized wild-type embeddings from the 11 protein mutation datasets. 
###    For each target protein, rank the other proteins according to their association scores. Note that the -k parameter should be set to 10.
python retrieve.py -m retrieve -md esm2 -b 16 -k 10 -mt cosine -cpu


# 2. meta-learning -> finetune -> test 
## train_size = 20
### 1) esm2 + meta + -mt=3 (train)
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=3 python main.py -md esm2 -m meta -ts 20 -tb 1 -r 16 -ls 5 -mi 5 -mtb 16 -meb 64 -alr 5e-3 -as 5 -e 100 -cv 5 -mt 3 -p all
### 2) esm2 + meta-transfer + -mt=3 (train)
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=3 python main.py -md esm2 -m meta-transfer -ts 20 -tb 16 -r 16 -ls 5 -mi 5 -mtb 16 -meb 64 -alr 5e-3 -as 5 -e 100 -cv 5 -mt 3 -p all
### 3) esm2 + meta-transfer + -mt=3 (test)
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=3 python main.py -md esm2 -m meta-transfer -ts 20 -tb 16 -r 16 -ls 5 -mi 5 -mtb 16 -meb 64 -alr 5e-3 -as 5 -e 100 -cv 5 -mt 3 -p all -t

## train_size = 40
### 1) esm2 + meta + -mt=3 (train)
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=3 python main.py -md esm2 -m meta -ts 40 -tb 1 -r 16 -ls 5 -mi 5 -mtb 16 -meb 64 -alr 5e-3 -as 5 -e 100 -cv 5 -mt 3 -p all
### 2) esm2 + meta-transfer + -mt=3 (train)
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=3 python main.py -md esm2 -m meta-transfer -ts 40 -tb 16 -r 16 -ls 5 -mi 5 -mtb 16 -meb 64 -alr 5e-3 -as 5 -e 100 -cv 5 -mt 3 -p all
### 3) esm2 + meta-transfer + -mt=3 (test)
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=3 python main.py -md esm2 -m meta-transfer -ts 40 -tb 16 -r 16 -ls 5 -mi 5 -mtb 16 -meb 64 -alr 5e-3 -as 5 -e 100 -cv 5 -mt 3 -p all -t

## train_size = 80
### 1) esm2 + meta + -mt=3 (train)
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=3 python main.py -md esm2 -m meta -ts 80 -tb 1 -r 16 -ls 5 -mi 5 -mtb 16 -meb 64 -alr 5e-3 -as 5 -e 100 -cv 5 -mt 3 -p all
### 2) esm2 + meta-transfer + -mt=3 (train)
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=3 python main.py -md esm2 -m meta-transfer -ts 80 -tb 16 -r 16 -ls 5 -mi 5 -mtb 16 -meb 64 -alr 5e-3 -as 5 -e 100 -cv 5 -mt 3 -p all
### 3) esm2 + meta-transfer + -mt=3 (test)
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=3 python main.py -md esm2 -m meta-transfer -ts 80 -tb 16 -r 16 -ls 5 -mi 5 -mtb 16 -meb 64 -alr 5e-3 -as 5 -e 100 -cv 5 -mt 3 -p all -t

## train_size = 160
### 1) esm2 + meta + -mt=3 (train)
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=3 python main.py -md esm2 -m meta -ts 160 -tb 1 -r 16 -ls 5 -mi 5 -mtb 16 -meb 64 -alr 5e-3 -as 5 -e 100 -cv 5 -mt 3 -p all
### 2) esm2 + meta-transfer + -mt=3 (train)
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=3 python main.py -md esm2 -m meta-transfer -ts 160 -tb 16 -r 16 -ls 5 -mi 5 -mtb 16 -meb 64 -alr 5e-3 -as 5 -e 100 -cv 5 -mt 3 -p all
### 3) esm2 + meta-transfer + -mt=3 (test)
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=3 python main.py -md esm2 -m meta-transfer -ts 160 -tb 16 -r 16 -ls 5 -mi 5 -mtb 16 -meb 64 -alr 5e-3 -as 5 -e 100 -cv 5 -mt 3 -p all -t

## train_size = 320
### 1) esm2 + meta + -mt=3 (train)
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=3 python main.py -md esm2 -m meta -ts 320 -tb 1 -r 16 -ls 5 -mi 5 -mtb 16 -meb 64 -alr 5e-3 -as 5 -e 100 -cv 5 -mt 3 -p all
### 2) esm2 + meta-transfer + -mt=3 (train)
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=3 python main.py -md esm2 -m meta-transfer -ts 320 -tb 16 -r 16 -ls 5 -mi 5 -mtb 16 -meb 64 -alr 5e-3 -as 5 -e 100 -cv 5 -mt 3 -p all
### 3) esm2 + meta-transfer + -mt=3 (test)
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=3 python main.py -md esm2 -m meta-transfer -ts 320 -tb 16 -r 16 -ls 5 -mi 5 -mtb 16 -meb 64 -alr 5e-3 -as 5 -e 100 -cv 5 -mt 3 -p all -t

# 3. summarize results
python summarize_results.py