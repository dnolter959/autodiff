#!/usr/bin/env bash

set -e

tool='coverage'
if [[ $# -gt 0 ]]; then
    # optional argument to use different tool to check coverage
    tool="${1}"; shift
fi

if [[ ${tool} == 'coverage' ]]; then
    # run the tests (generates coverage data to build report)
    cd tests && ./run_tests.sh coverage run --source=../src "${@}"

    # build the coverage report on stdout
    coverage report -m
elif [[ ${tool} == 'pytest' ]]; then
    # generate coverage reports with pytest in one go
    cd tests && ./run_tests.sh pytest --cov=../src --cov-report term --cov-report=html:htmlcov "${@}" --cov-fail-under=90
else
    # error: write to stderr
    >&2 echo "Error: unknown tool '${tool}'"
    exit 1
fi
