CUDA_VISIBLE_DEVICES=0 python TTS/bin/train_tts.py \
	--config_path config_yourtts_with_brspeech.json \
	--restore_path ./checkpoints/vits_tts_mls-May-21-2022_09+23PM-2d64d351/checkpoint_1210000.pth.tar

