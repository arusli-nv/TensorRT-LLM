#!/bin/bash
# Runs the command as a child and exits with its status, as trtllm-llmapi-launch does.
"$@"
exit $?
