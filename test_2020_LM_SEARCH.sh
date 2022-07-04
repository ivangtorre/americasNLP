#########################################################################################################
###
checkpoint="/home/ivan/Iberspeech2022/outputs/2022-06-26/07-28-52/checkpoints/checkpoint_best.pt"
data="/data/IBERSPEECH2022/"
lm_model="/data/IBERSPEECH2022/TEXT_CORPUS/subtitles2018/3gram.binary"
lexicon="/data/IBERSPEECH2022/LEXICON/google_ngram/lexicon.txt"
test_name="test_2020"
#
#
#
##############################################################################################
#nvidia-docker run -it \
docker run -it --rm -e NVIDIA_VISIBLE_DEVICES=none \
--network=host --name "fairseq_test" --shm-size=4g --ulimit memlock=-1 \
-v /home/:/home/ -v /data/:/data/ -w /workspace/fairseq \
fairseq \
python3 examples/speech_recognition/new/infer.py --multirun \
--config-path=/home/ivan/Iberspeech2022/hydra_search_lm/config1 \
hydra/sweeper=ax \
task=audio_finetuning \
common_eval.path=$checkpoint \
decoding.type=kenlm \
decoding.lexicon=$lexicon \
decoding.lmpath=$lm_model \
dataset.gen_subset=$test_name
common.cpu=True





