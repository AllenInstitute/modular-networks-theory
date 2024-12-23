#!/usr/bin/env bash
set -ex

## Uncomment for faster results in the debug mode (with smaller networks and shorter simulations):
#python analysis_nested_stats.py -s 3.5 -d
#python analysis_nested_stats.py -s 8.0 -m --no-legend -d

python analysis_nested_stats.py -s 3.5 
python analysis_nested_stats.py -s 8.0 -m --no-legend 