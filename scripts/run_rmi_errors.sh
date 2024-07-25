#!bash
# set -x
trap "exit" SIGINT

EXPERIMENT="rmi errors"

DIR_DATA="data"
DIR_RESULTS="results"
FILE_RESULTS="${DIR_RESULTS}/rmi_errors.csv"

BIN="build/bin/rmi_errors"

run() {
    DATASET=$1
    LAYER1=$2
    LAYER2=$3
    N_MODELS=$4
    SWITCH=$5
    DATA_FILE="${DIR_DATA}/${DATASET}"
    ${BIN} ${DATA_FILE} ${LAYER1} ${LAYER2} ${N_MODELS} ${SWITCH} >> ${FILE_RESULTS}
}

# Create results directory
if [ ! -d "${DIR_RESULTS}" ];
then
    mkdir -p "${DIR_RESULTS}";
fi

# Check data downloaded
if [ ! -d "${DIR_DATA}" ];
then
    >&2 echo "Please download datasets first."
    return 1
fi

DATASETS="books_200M_uint64 fb_200M_uint64 osm_cellids_200M_uint64 wiki_ts_200M_uint64"
LAYER1_MODELS="linear_spline"
LAYER2_MODELS="linear_regression linear_regression_welford linear_regression_welford_float"

# Run experiments
echo "dataset,n_keys,layer1,layer2,n_models,size_in_bytes,switch_n,mean_ae,median_ae,stdev_ae,min_ae,max_ae" > ${FILE_RESULTS} # Write csv header
for dataset in ${DATASETS};
do
    echo "Performing ${EXPERIMENT} on '${dataset}'..."
    for ((i=6; i<=27; i += 3));
    do
        n_models=$((2**$i))
        for l1 in ${LAYER1_MODELS};
        do
            for l2 in ${LAYER2_MODELS};
            do
                if [[ "${l2}" == "linear_regression_welford_float" ]]; then
                        start_j=16
                        stop_j=16
                    else
                        start_j=8
                        stop_j=8
                    fi
                    
                    for ((j=start_j; j<=stop_j; j += 8));
                    do
                        SWITCH=$j
                        run ${dataset} ${l1} ${l2} ${n_models} ${SWITCH}
            done
        done
    done
done
done
