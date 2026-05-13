SHELL := /bin/bash
PYTHON ?= python

.PHONY: setup prepare features smoke train train-low-resource ml-baselines lambda-ablation analysis paper-tables summary teacher-predictions-regression teacher-predictions-multiteacher teacher-reliability distill-regression distill-regression-summary distill-lambda-sweep distill-lambda-summary distill-lambda-analysis distill-adaptive-smoke distill-adaptive-regression distill-adaptive-summary

setup:
	bash setup_env.sh

prepare:
	$(PYTHON) prepare_data.py --zip-path /Users/yuzibo/Downloads/admet_group.zip

features:
	$(PYTHON) train.py --generate-features --features-only --datasets bbb_martins caco2_wang --train-ratio-tags 50

smoke:
	$(PYTHON) train.py --generate-features --datasets bbb_martins --models ECFP4_MLP --seeds 42 --train-ratio-tags 50

train:
	$(PYTHON) train.py --generate-features

train-low-resource:
	$(PYTHON) train.py --generate-features --train-ratio-tags 10 20 50 --skip-existing

ml-baselines:
	$(PYTHON) train_ml_baselines.py --train-ratio-tags 10 20 50 --skip-existing

lambda-ablation:
	$(PYTHON) train.py --datasets bbb_martins caco2_wang --models ECFP4_MLP_DescPred --seeds 42 123 3407 --lambda-transfer 0.01 --results-root results_lambda/lambda_0_01
	$(PYTHON) train.py --datasets bbb_martins caco2_wang --models ECFP4_MLP_DescPred --seeds 42 123 3407 --lambda-transfer 0.1 --results-root results_lambda/lambda_0_1
	$(PYTHON) train.py --datasets bbb_martins caco2_wang --models ECFP4_MLP_DescPred --seeds 42 123 3407 --lambda-transfer 1.0 --results-root results_lambda/lambda_1_0

analysis:
	$(PYTHON) analyze_adapter_fusion.py

paper-tables:
	$(PYTHON) build_paper_tables.py
	$(PYTHON) build_distillation_paper_tables.py

summary:
	$(PYTHON) evaluate.py

teacher-predictions-regression:
	$(PYTHON) generate_teacher_predictions.py --datasets caco2_wang lipophilicity_astrazeneca solubility_aqsoldb vdss_lombardo ppbr_az --teacher-model ECFP4_Desc_RF --train-ratio-tags 10 20 50 --seeds 42 123 3407 --skip-existing

teacher-predictions-multiteacher:
	$(PYTHON) generate_teacher_predictions.py --datasets caco2_wang lipophilicity_astrazeneca solubility_aqsoldb vdss_lombardo ppbr_az --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF ECFP4_XGB Desc_XGB ECFP4_Desc_XGB --train-ratio-tags 10 20 50 --seeds 42 123 3407 --skip-existing

teacher-reliability:
	$(PYTHON) analyze_teacher_reliability.py

distill-regression:
	$(PYTHON) train.py --datasets caco2_wang lipophilicity_astrazeneca solubility_aqsoldb vdss_lombardo ppbr_az --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 20 50 --seeds 42 123 3407 --teacher-root data/teacher_predictions --teacher-model ECFP4_Desc_RF --lambda-distill 0.1 --results-root results_distill_regression --skip-existing

distill-regression-summary:
	$(PYTHON) evaluate.py --results-root results_distill_regression
	$(PYTHON) compare_distillation.py --distill-results-root results_distill_regression

distill-lambda-sweep:
	$(PYTHON) train.py --datasets caco2_wang lipophilicity_astrazeneca solubility_aqsoldb vdss_lombardo ppbr_az --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 20 50 --seeds 42 123 3407 --teacher-root data/teacher_predictions --teacher-model ECFP4_Desc_RF --lambda-distill 0.01 --results-root results_distill_lambda/lambda_0_01 --skip-existing
	$(PYTHON) train.py --datasets caco2_wang lipophilicity_astrazeneca solubility_aqsoldb vdss_lombardo ppbr_az --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 20 50 --seeds 42 123 3407 --teacher-root data/teacher_predictions --teacher-model ECFP4_Desc_RF --lambda-distill 0.3 --results-root results_distill_lambda/lambda_0_3 --skip-existing
	$(PYTHON) train.py --datasets caco2_wang lipophilicity_astrazeneca solubility_aqsoldb vdss_lombardo ppbr_az --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 20 50 --seeds 42 123 3407 --teacher-root data/teacher_predictions --teacher-model ECFP4_Desc_RF --lambda-distill 1.0 --results-root results_distill_lambda/lambda_1_0 --skip-existing

distill-lambda-summary:
	$(PYTHON) compare_distillation_lambdas.py

distill-lambda-analysis:
	$(PYTHON) analyze_distillation_lambdas.py

distill-adaptive-smoke:
	$(PYTHON) train.py --datasets caco2_wang --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 --seeds 42 --teacher-root data/teacher_predictions --teacher-model ECFP4_Desc_RF --lambda-distill 1.0 --adaptive-distill-strategy teacher_valid_advantage --adaptive-base-results-root results --results-root results_distill_adaptive_smoke

distill-adaptive-regression:
	$(PYTHON) train.py --datasets caco2_wang lipophilicity_astrazeneca solubility_aqsoldb vdss_lombardo ppbr_az --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 20 50 --seeds 42 123 3407 --teacher-root data/teacher_predictions --teacher-model ECFP4_Desc_RF --lambda-distill 1.0 --adaptive-distill-strategy teacher_valid_advantage --adaptive-base-results-root results --results-root results_distill_adaptive_regression --skip-existing

distill-adaptive-summary:
	$(PYTHON) compare_adaptive_distillation.py
	$(PYTHON) analyze_distillation_lambdas.py --lambda-root adaptive=results_distill_adaptive_regression --output-root results_distill_adaptive_regression/analysis
