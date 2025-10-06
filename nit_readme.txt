CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 bash scripts/train/stage4_2b_finetune.sh 1 6

bash scripts/train/stage4_2b_finetune.sh 1 8

 nohup bash scripts/train/stage4_2b_finetune.sh 1 8 > logs/stage4_2b_finetune_chunksv0.log 2>&1 &