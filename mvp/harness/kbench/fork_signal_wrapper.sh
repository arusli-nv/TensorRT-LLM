#!/bin/bash
# As trtllm-llmapi-launch (worker branch): runs the command as a child and ends by its signal.
"$@"
code=$?
if [ "$code" -gt 128 ]; then kill -s $((code - 128)) $$; fi
exit $code
