#!/bin/bash
# Benchmark Script: Run N qualification matches and log capture times

N=${1:-5}
echo "=== Running $N Benchmark Qualification Matches ==="

LOG_FILE="benchmark_results_$(date +%Y%m%d_%H%M%S).log"
echo "Logging results to $LOG_FILE"

for i in $(seq 1 $N); do
    echo "--- Match $i of $N ---" | tee -a "$LOG_FILE"
    START_TIME=$(date +%s)
    
    # Launch simulation and catcher with timeout
    timeout 185 ros2 launch turtlebot_pe catcher.launch.py &
    CATCHER_PID=$!
    
    wait $CATCHER_PID 2>/dev/null
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    echo "Match $i ended after $ELAPSED seconds." | tee -a "$LOG_FILE"
    sleep 3
done

echo "=== Benchmark Complete ===" | tee -a "$LOG_FILE"
