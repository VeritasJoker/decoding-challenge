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
CMD := python
CMD := sbatch submit.sh

# grid (6v for 1 & 2, BA44 for 3 & 4)
%-model: GRID := 6v

# neural feature (spike, tx1, tx2, tx3, tx4)
%-model: NF := spike

# model_size
%-model: MODEL_SIZE := tiny.en


build-model:
	python scripts/model_build.py \
		--grid $(GRID) \
		--feature $(NF) \
		--model-size $(MODEL_SIZE) \
		--saving-dir $(MODEL_SIZE)-$(GRID)-$(NF); \


train-model:
	$(CMD) scripts/model_train2.py \
		--project-id $(PRJCT_ID) \
		--sid $(SID) \
		--model-size $(MODEL_SIZE) \
		--data-split $(DATA_SPLIT) \
		--data-split-type $(DATA_SPLIT_TYPE) \
		--elec-type $(ELEC_TYPE) \
		--onset-shift $(ONSET_SHIFT) \
		--ecog-type $(ECOG_TYPE) \
		--seg-type $(SEG_TYPE) \
		--saving-dir whisper-$(MODEL_SIZE)-$(SID)-$(ELEC_TYPE)-$(DATA_SPLIT_TYPE)-test$(DATA_SPLIT); \

