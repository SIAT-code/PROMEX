# 1. processing data
### 1) Process the wild-type sequences and the DataFrame
python preprocess.py
### 2) Vectorize the wild-type sequences
CUDA_VISIBLE_DEVICES=2 python retrieve.py -m vectorize -md esm2 -b 8
### 3) Compute an association matrix using the vectorized wild-type embeddings from the 11 protein mutation datasets. 
###    For each target protein, rank the other proteins according to their association scores.
###    Note that set k = 10 while in round1, k = 2 whild in round2.
python retrieve.py -m retrieve -md esm2 -b 16 -k 10 -mt cosine -cpu

# 2. meta-learning -> finetune -> test 
### 1. Meta-train PLMs
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=2 python main.py -md promex -m meta -ts 0 -tb 1 -r 16 -ls 3 -mi 5 -mt 4 -mtb 16 -meb 64 -alr 5e-3 -as 5 -e 100 -cv 5 -p all
### 2. Transfer the meta-trained model to the target task
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=2 python main.py -md promex -m meta-transfer -ts 0 -tb 16 -r 16 -ls 3 -mi 5 -mt 4 -mtb 16 -meb 64 -alr 5e-3 -as 5 -e 100 -cv 5 -p all
### 3. Test the trained model, print results, and save predictions
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=2 python main.py -md promex -m meta-transfer -ts 0 -tb 16 -r 16 -ls 3 -mi 5 -mt 4 -mtb 16 -meb 64 -alr 5e-3 -as 5 -e 100 -cv 5 -p all -t





