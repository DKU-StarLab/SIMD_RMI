#!bash
# set -x
trap "exit" SIGINT

EXPERIMENT="rmi build"

DIR_DATA="data"
DIR_RESULTS="results"
FILE_RESULTS="${DIR_RESULTS}/rmi_build.csv"

BIN="build/bin/rmi_build"

# Set number of repetitions and samples
N_REPS="1"
PARAMS="--n_reps ${N_REPS}"
TIMEOUT="60s"

DATASETS="books_200M_uint64 fb_200M_uint64 osm_cellids_200M_uint64 wiki_ts_200M_uint64"
LAYER1="linear_spline"
LAYER2="linear_regression linear_regression_welford linear_regression_welford_float"
BOUNDS="none"

run() {
    DATASET=$1
    L1=$2
    L2=$3
    N_MODELS=$4
    BOUND=$5
    SWITCH=$6
    DATA_FILE="${DIR_DATA}/${DATASET}"
    timeout ${TIMEOUT} ${BIN} ${DATA_FILE} ${L1} ${L2} ${N_MODELS} ${BOUND} ${PARAMS} ${SWITCH} >> ${FILE_RESULTS}
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

# Write csv header
echo "dataset,n_keys,rmi,layer1,layer2,n_models,bounds,size_in_bytes,switch_n,rep,build_time,tran_time,err_time,checksum" > ${FILE_RESULTS} # Write csv header

# Run layer1 and layer 2 model type experiment
for dataset in ${DATASETS};
do
    echo "Performing ${EXPERIMENT} (ours) on '${dataset}'..."
    for ((i=6; i<=27; i += 3));
    do
        n_models=$((2**$i))
        for l1 in ${LAYER1};
        do
            for l2 in ${LAYER2};
            do
                for bound in ${BOUNDS};
                do
                    if [[ "${l2}" == "linear_regression" ]]; then
                        run ${dataset} ${l1} ${l2} ${n_models} ${bound} 8
                    fi
                    if [[ "${l2}" == "linear_regression_welford" ]]; then
                        start_j=8
                        stop_j=64
                        for ((j=start_j; j<=stop_j; j += 8));
                        do
                            switch=$j
                            run ${dataset} ${l1} ${l2} ${n_models} ${bound} ${switch}
                        done
                    fi
                    if [[ "${l2}" == "linear_regression_welford_float" ]]; then
                        start_j=16
                        stop_j=128
                        for ((j=start_j; j<=stop_j; j += 16));
                        do
                            switch=$j
                            run ${dataset} ${l1} ${l2} ${n_models} ${bound} ${switch}
                        done
                    fi
                done
            done
        done
    done
done
