# !/bin/bash

for i in {0..2}
do
#    mkdir data$i/checkpoint
    mv data$i/train_actions_training data$i/train/data.trgt
    mv data$i/train_instructions_training data$i/train/data.srce
    mv data$i/train_map_training data$i/train/map.srce
    mv data$i/train_positions_training data$i/train/positions.trgt

    mv data$i/train_actions_validation data$i/dev/data.trgt
    mv data$i/train_instructions_validation data$i/dev/data.srce
    mv data$i/train_map_validation data$i/dev/map.srce
    mv data$i/train_positions_validation data$i/dev/positions.trgt

    mv data$i/test_action data$i/test/data.trgt
    mv data$i/test_instructions data$i/test/data.srce
    mv data$i/test_map data$i/test/map.srce
    mv data$i/test_positions data$i/test/positions.trgt

#    mkdir data$i/train
#    mkdir data$i/dev
#    mkdir data$i/test
done
