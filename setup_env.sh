#!/usr/bin/env bash
readonly SCRIPT_DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd);
readonly ENV_NAME=tf-fs

function main
{
    echo "[*] Install fsl conda environment.";

    set -o errexit
    set -o pipefail
    set -o nounset
    set -o errtrace

    install_env;
}

function install_env
{
    echo "[*] Installing dependencies...";
    if conda env list | grep -q ${ENV_NAME}
    then
        echo "[*] ${ENV_NAME} env found, updating...";
        conda env update -q -f ${SCRIPT_DIRECTORY}/environment.yml -n ${ENV_NAME};
    else
        echo "[*] Creating ${ENV_NAME} env...";
        conda env create -q -f ${SCRIPT_DIRECTORY}/environment.yml -n ${ENV_NAME};
    fi

    echo "[*] Activating ${ENV_NAME}..";
    PS1=${PS1:-}
    source activate ${ENV_NAME};

    echo "[*] Done !";
}

main
