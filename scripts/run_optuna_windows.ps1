# NOTE: Archival wrapper -- uses hardcoded PSC paths under data/out/.
# Not domain-aware. For domain-isolated runs, invoke optuna_msd_search.py
# directly with --domain <domain_id>.
$windows = @(
    @{lagMin=8; lagMax='none'; study='msd_meta_tuning_cat_lag8inf'},
    @{lagMin=6; lagMax='12';  study='msd_meta_tuning_cat_lag6_12'},
    @{lagMin=8; lagMax='16'; study='msd_meta_tuning_cat_lag8_16'},
    @{lagMin=4; lagMax='12'; study='msd_meta_tuning_cat_lag4_12'},
    @{lagMin=8; lagMax='20'; study='msd_meta_tuning_cat_lag8_20'}
)

foreach ($win in $windows) {
    $lagMaxArg = $win.lagMax

    python scripts/msd_meta_tune.py `
      --labels data/out/02_lineage_tracking/inflection_labels.csv `
      --lag-min $win.lagMin `
      --lag-max $lagMaxArg `
      --model-types catboost `
      --n-trials 40 `
      --cv-folds 3 `
      --optuna-n-jobs 4 `
      --estimator-n-jobs 16 `
      --study-name $win.study `
      --storage "sqlite:///data/out/experiments/$($win.study)/optuna.db" `
      --output-dir "data/out/experiments/$($win.study)" `
      --prune
}
