docker run -it --rm --network=host --name "fairseq_quechua" --shm-size=4g --ulimit memlock=-1 \
-v /home/:/home/ -v /data/:/data/ -w $PWD fairseq fairseq-hydra-train --config-dir config \
--config-name finetune_quechua.yaml
#nvidia-docker run -it --rm --network=host --name "fairseq_quechua" --shm-size=4g --ulimit memlock=-1 \
#-v /home/:/home/ -v /data/:/data/ -w $PWD fairseq fairseq-hydra-train --config-dir config \
#--config-name finetune_qeuchua.yaml



