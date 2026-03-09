python scripts/multi_signal_detector.py \
    --labels data/out/02_lineage_tracking/inflection_labels.csv \
    --multisignal data/out/02_lineage_tracking/lineage_multisignal_features.csv \
    --tight-mapping data/out/experiments/stage0_tight_mapping/milestone_lineage_mapping_tight.csv \
    --semantic-velocity data/out/experiments/stage1_quarterly_embeddings/semantic_velocity.csv \
    --timeseries data/out/02_lineage_tracking/lineage_timeseries.csv \
    --output-dir data/out/experiments/msd_inflection \
    --model lightgbm --use-cv --cv-folds 5 --threshold 0.3
