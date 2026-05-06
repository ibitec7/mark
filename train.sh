#!/bin/bash

# Assert all the necessary configs exist:
FILE_PATHS=("./configs/training_config_chebyshev_stage1.yaml" "./configs/training_config_chebyshev_stage2.yaml"\
 "./configs/training_config_chebyshev_stage3.yaml" "./configs/training_config_hypernet_stage1.yaml" \
 "./configs/training_config_hypernet_stage2.yaml" "./configs/training_config_hypernet_stage3.yaml" \
 "./configs/training_config_dct_stage1.yaml" "./configs/training_config_dct_stage2.yaml" \
 "./configs/training_config_dct_stage3.yaml")

for file in "${FILE_PATHS[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Required config file $file not found!"
        exit 1
    fi
done

# First we have all our pretrained weights trainsfered for training.
cp ./configs/training_config_chebyshev_stage1.yaml ./configs/training_config.yaml &&
python -m src.transfer;

cp ./configs/training_config_hypernet_stage1.yaml ./configs/training_config.yaml &&
python -m src.transfer;

cp ./configs/training_config_dct_stage1.yaml ./configs/training_config.yaml &&
python -m src.transfer;

# Now we load our relevant configs and train in 3 stages each.
cp ./configs/training_config_chebyshev_stage1.yaml ./configs/training_config.yaml &&
python -m src.main &&
cp -r ./checkpoints/hydra_mark ./checkpoints/hydra_mark_chebyshev_stage1 &&
cp ./configs/training_config_chebyshev_stage2.yaml ./configs/training_config.yaml &&
python -m src.main &&
cp -r ./checkpoints/hydra_mark ./checkpoints/hydra_mark_chebyshev_stage2 &&
cp ./configs/training_config_chebyshev_stage3.yaml ./configs/training_config.yaml &&
python -m src.main &&
cp -r ./checkpoints/hydra_mark ./checkpoints/hydra_mark_chebyshev_stage3;

cp ./configs/training_config_hypernet_stage1.yaml ./configs/training_config.yaml &&
python -m src.main &&
cp -r ./checkpoints/hydra_mark ./checkpoints/hydra_mark_hypernet_stage1 &&
cp ./configs/training_config_hypernet_stage2.yaml ./configs/training_config.yaml &&
python -m src.main &&
cp -r ./checkpoints/hydra_mark ./checkpoints/hydra_mark_hypernet_stage2 &&
cp ./configs/training_config_hypernet_stage3.yaml ./configs/training_config.yaml &&
python -m src.main
cp -r ./checkpoints/hydra_mark ./checkpoints/hydra_mark_hypernet_stage3;

cp ./configs/training_config_dct_stage1.yaml ./configs/training_config.yaml &&
python -m src.main &&
cp -r ./checkpoints/hydra_mark ./checkpoints/hydra_mark_dct_stage1 &&
cp ./configs/training_config_dct_stage2.yaml ./configs/training_config.yaml &&
python -m src.main &&
cp -r ./checkpoints/hydra_mark ./checkpoints/hydra_mark_dct_stage2 &&
cp ./configs/training_config_dct_stage3.yaml ./configs/training_config.yaml &&
python -m src.main &&
cp -r ./checkpoints/hydra_mark ./checkpoints/hydra_mark_dct_stage3;