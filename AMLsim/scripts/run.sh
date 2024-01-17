#!/usr/bin/env bash


if [[ $# -ne 1 ]]; then
    echo "Usage: sh $0 [ConfJSON]"
    exit 1
fi

MIN_HEAP=2g
MAX_HEAP=4g

CONF_JSON=$1

# get directory name from conf.json
DIR=$(jq -r '.input.directory' "${CONF_JSON}")

# get degree distribution file from conf.json
DEGREE=$(jq -r '.input.degree' "${CONF_JSON}")

# check if degree distribution file exists
if [[ ! -f "${DIR}/${DEGREE}" ]]; then
    echo "degree distribution file not found: ${DIR}/${DEGREE}"
    echo "creating degree distribution file..."
    # create degree distribution file
    python3 scripts/generate_scalefree.py "${CONF_JSON}"
    echo "done"
fi

echo "generating transaction graph..."
python3 scripts/transaction_graph_generator.py "${CONF_JSON}"
echo "done"

if ! command -v mvn
then
    echo 'maven not installed. proceeding.'
    java -XX:+UseConcMarkSweepGC -XX:ParallelGCThreads=2 -Xms${MIN_HEAP} -Xmx${MAX_HEAP} -cp "jars/*:target/classes/" amlsim.AMLSim "${CONF_JSON}"
    exit
else
    echo 'maven is installed. proceeding'
    mvn exec:java -Dexec.mainClass=amlsim.AMLSim -Dexec.args="${CONF_JSON}"
fi

# Cleanup temporal outputs of AMLSim
rm -f outputs/_*.csv outputs/_*.txt outputs/summary.csv
