CUDA_VISIBLE_DEVICES=5 python run.py --outf model1  --margin 100  --manualSeed 1 --lr 0.01 --batch-size 256
CUDA_VISIBLE_DEVICES=6 python run.py --outf model2  --margin 1000 --manualSeed 1 --lr 0.01 --batch-size 256

CUDA_VISIBLE_DEVICES=7 python run.py --outf model1  --margin 100  --manualSeed 1 --lr 0.05 --batch-size 1024 --mode vis
