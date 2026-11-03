#!/bin/bash
# Gate a GPU job on a finished build job's unit tests: every "rc=" result in the build's Slurm log is rc=0 and the
# named last test file ran. Usage: require_build_pass.sh BUILD_SLURM_OUT LAST_TEST_FILE [EXTRA_TEST_LOG]
# With KNOWN_FAIL=<pytest id> and the build's test-extra.log, one "extra <file> rc=1" is accepted when the only FAILED
# line in that log is that id (FREEZE_CHECKLIST C12: test_remote_mpi_session[flashinfer_temporary_cleanup]).
log=$1 last=$2 extra_log=${3:-}
grep -q "^extra $last rc=0" "$log" || { echo "BUILD_GATE_FAIL: $last did not pass in $log"; exit 1; }
bad=$(grep -E "rc=[^0]|rc=[0-9]{2,}" "$log" | grep -v "rc=0\b")
if [[ -n ${KNOWN_FAIL:-} && -n $bad && -n $extra_log ]]; then
    failed=$(grep -E "^FAILED " "$extra_log" | sed 's/ - .*//' | sort -u)
    if [[ $(echo "$bad" | wc -l) == 1 && $bad == "extra "*" rc=1" && $(echo "$failed" | wc -l) == 1 \
          && $failed == *"$KNOWN_FAIL"* ]]; then
        echo "BUILD_GATE_KNOWN: $bad ($failed)"; bad=
    fi
fi
[[ -z $bad ]] || { echo "BUILD_GATE_FAIL: $(echo "$bad" | head -3)"; exit 1; }
echo "BUILD_GATE_OK $log"
