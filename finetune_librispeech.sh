## MONITORIZE GPU
#nvidia-smi --query-gpu=memory.used --format=csv --loop=5 --filename=gpu_utillization.csv &

nvidia-docker run -it --rm --network=host --name "fairseq_container" --shm-size=4g --ulimit memlock=-1 \
-v /home/:/home/ -v /data/:/data/ -w $PWD fairseq fairseq-hydra-train --config-dir config \
--config-name finetune_librispeech.yaml

## FINISH MONITORIZATION AND GET MAX USAGE
#ps aux  |  grep -i "nvidia-smi --query-gpu=memory" |  awk '{print $2}'  | xargs kill -15
#cat gpu_utillization.csv | sort -rn | head -n 1 > max_memory_gpu.txt
#cat max_memory_gpu.txt


