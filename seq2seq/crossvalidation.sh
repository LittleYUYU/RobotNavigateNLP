#!/bin/bash


# Usage:
# bash crossvalidation.sh geo/ifttt/other size keep_prob learning_rate num_layers

function train(){
    data_name=$1
    data_dir=$2
    size=$3
    keep_prob=$4
    learning_rate=$5
    num_layers=$6
    outputFile=$7

    python seq2seq_run.py --data_name=$data_name --data_dir=$data_dir --size=$size --keep_prob=$keep_prob --learning_rate=$learning_rate --num_layers=$num_layers 2>&1 > $outputFile
}

function decode(){
    data_name=$1
    data_dir=$2
    size=$3
    keep_prob=$4
    learning_rate=$5
    num_layers=$6

    python seq2seq_run.py --data_name=$data_name --data_dir=$data_dir --size=$size --keep_prob=$keep_prob --learning_rate=$learning_rate --num_layers=$num_layers --decode=True
}


if [ $# != 5 ]
    then
        echo "Wrong argument size. Exit."
        exit
fi

printf "\n============Cross validation on %s============\n" "$1"


for i in {0..9}
do
#    if [[ $i -eq 3 || $i -eq 7 ]]
#    then
#        rm -r ../data/$1/data$i/checkpoint
#        mkdir ../data/$1/data$i/checkpoint
#        printf "\nCross validation on data%s\n" $i
#        printf "training...\n"
#        train $1 ../data/$1/data$i $2 $3 $4 $5 ../data/$1/data$i/$2"_"$3"_"$4"_"$5".out"
#        printf "decode...\n"
#        decode $1 ../data/$1/data$i
#        printf "Done\n\n"
#    fi
    rm -r ../data/$1/data$i/checkpoint
    mkdir ../data/$1/data$i/checkpoint
    printf "\nCross validation on data%s\n" $i
    printf "training...\n"
    train $1 ../data/$1/data$i $2 $3 $4 $5 ../data/$1/data$i/$2"_"$3"_"$4"_"$5".out"
    printf "decode...\n"
    decode $1 ../data/$1/data$i $2 $3 $4 $5
    printf "Done\n\n"
done

