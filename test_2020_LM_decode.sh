# ######### VARIABLES ############################
TEST_NAME=${TEST_NAME:-"test_2020"}
RESULT_PATH=${RESULT_PATH:-"/home/ivan/Iberspeech2022/outputs/"}
LEXICON=${LEXICON:-"/data/IBERSPEECH2022/LEXICON/google_ngram/lexicon.txt"}
##############################################
############# MODEL  #####################
#
#
MODEL=${MODEL:-"/home/ivan/Iberspeech2022/outputs/2022-06-26/07-28-52/checkpoints/checkpoint_best.pt"}
LM_MODEL=${LM_MODEL:-"/data/IBERSPEECH2022/TEXT_CORPUS/subtitles2018/3gram.binary"}
LM_WEIGHT=${LM_WEIGHT:-"0.8896091388773608"}
WORD_SCORE=${WORD_SCORE:-"0.8891034348782743"}
SIL_WEIGHT=${SIL_WEIGHT:-"-0.7867147781553445"}
BEAM=${BEAM:-"256"}
#BEAM=${BEAM:-"50"}
#
#
########################################

docker run -it --rm -e NVIDIA_VISIBLE_DEVICES=none --network=host --name "fairseq_test" --shm-size=4g --ulimit memlock=-1 \
-v /home/:/home/ -v /data/:/data/ -w /workspace/fairseq fairseq \
python3 examples/speech_recognition/infer.py \
/data/IBERSPEECH2022/ \
--gen-subset ${TEST_NAME} \
--path ${MODEL} \
--results-path ${RESULT_PATH} \
--lexicon ${LEXICON} \
--w2l-decoder kenlm \
--lm-model ${LM_MODEL} \
--lm-weight ${LM_WEIGHT} \
--word-score ${WORD_SCORE} \
--beam ${BEAM} \
--sil-weight ${SIL_WEIGHT} \
--cpu \
--scoring wer \
--batch-size 1 \
--num-workers 8 \
--task audio_finetuning \
--nbest 1 \
--post-process letter \
--criterion ctc \
--labels ltr \
--max-tokens-valid 40000000 \
--max-sample-size 560000 \
--min-sample-size 16000 \
--eval-wer