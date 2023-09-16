##########################################################
############## None Configurable Parameters ##############
##########################################################
USR := $(shell whoami | head -c 2)




##########################################################
######################### MODEL ##########################
##########################################################
# Choose the command to run: python runs locally, echo is for debugging, sbatch
# is for running on SLURM
CMD := echo
CMD := sbatch submit.sh
CMD := python

# grid (6v for 1 & 2, BA44 for 3 & 4)
%-model: GRID := all
%-model: GRID := 6v

# neural feature (spike, tx1, tx2, tx3, tx4)
%-model: NF := spikePow

# model_size
%-model: MODEL_SIZE := tiny.en


build-model:
	python scripts/model_build.py \
		--grid $(GRID) \
		--feature $(NF) \
		--model-size $(MODEL_SIZE) \
		--saving-dir $(MODEL_SIZE)-$(GRID)-$(NF); \


train-model:
	$(CMD) scripts/model_trainer.py \
		--grid $(GRID) \
		--feature $(NF) \
		--model-size $(MODEL_SIZE) \
		--saving-dir $(MODEL_SIZE)-$(GRID)-$(NF); \

