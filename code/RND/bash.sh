CUDA_VISIBLE_DEVICES=0 python train.py --use-gae --use-gpu --sticky-action --env-id SeaquestNoFrameskip-v4 &
CUDA_VISIBLE_DEVICES=1 python train.py --use-gae --use-gpu --sticky-action --env-id MontezumaRevengeNoFrameskip-v4 &
CUDA_VISIBLE_DEVICES=2 python train.py --use-gae --use-gpu --sticky-action --env-id RoadRunnerNoFrameskip-v4 & 
CUDA_VISIBLE_DEVICES=3 python train.py --use-gae --use-gpu --sticky-action --env-id BattleZoneNoFrameskip-v4 &