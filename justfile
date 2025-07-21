train-gpu +args: docker-build
    docker run --gpus device=0 --rm \
    -v $(pwd):/TransFusion \
    -w /TransFusion \
    transfusion \
    python train.py {{args}}

train +args: docker-build
    docker run --rm \
    -v $(pwd):/TransFusion \
    -w /TransFusion \
    transfusion \
    python train.py {{args}}

docker-build:
    docker build -t transfusion docker-image