echo "Start time:"
TZ=CET date
echo ""

DATA_DIR="data/cls-acl10-unprocessed"


# if data wasn't downloaded before
if [ ! -d "${DATA_DIR}/nl/books" ] ; then
    echo "Downloading 110kDBRD dataset"
    url="https://github.com/benjaminvdb/110kDBRD/releases/download/v2.0/110kDBRD_v2.tgz"
    wget -O ${DATA_DIR}/110kDBRD_v2.tgz ${url}

    # check if download failed
    if [ $? -ne 0 ]; then
        echo "Download failed. Check if the data set is still hosted at ${url}"
        exit
    fi
    echo "Unpacking tgz file"
    tar zxvf ${DATA_DIR}/110kDBRD_v2.tgz -C ${DATA_DIR}

    mkdir ${DATA_DIR}/nl
    mkdir ${DATA_DIR}/nl/books

    # convert text files labeled folders into just one text file with labels per split
    python convert_110kDBRD_laser.py
else
    echo "Dataset already exists. Skipping download. "
fi