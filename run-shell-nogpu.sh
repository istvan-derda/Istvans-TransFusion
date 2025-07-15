#!/bin/bash

docker run -it --rm \
    -v $(pwd)/../:/home/$USER \
    transfusion \
    /bin/bash -c "cd /home/$USER; exec /bin/bash"
