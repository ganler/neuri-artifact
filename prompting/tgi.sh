#!/bin/bash

# create bash arguments for "tgi_home", "gpus", and "max_tokens"
while getopts h:g:t: option; do
    case "${option}" in
        h) TGI_HOME=${OPTARG};;
        g) GPUS=${OPTARG};;
        t) MAX_TOKENS=${OPTARG};;
        \?) echo "Invalid option: -$OPTARG" >&2; exit 1;;
    esac
done

# assert these options are set
if [ -z "$TGI_HOME" ]; then
    echo "TGI_HOME is unset"
    exit 1
fi

if [ -z "$GPUS" ]; then
    echo "GPUS is unset"
    exit 1
fi

if [ -z "$MAX_TOKENS" ]; then
    echo "MAX_TOKENS is unset"
    exit 1
fi

model=TheBloke/deepseek-coder-33B-instruct-AWQ
volume=$TGI_HOME/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus "device=${GPUS}" --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.3 --model-id $model --quantize awq \
                          --max-total-tokens $MAX_TOKENS
