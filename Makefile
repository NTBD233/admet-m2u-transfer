SHELL := /bin/bash
PYTHON ?= python

.PHONY: setup prepare features smoke train train-low-resource ml-baselines lambda-ablation analysis paper-tables summary teacher-predictions-regression teacher-predictions-multiteacher teacher-reliability gate-target-diagnostics multiteacher-uniform-smoke multiteacher-validation-smoke multiteacher-top1-smoke multiteacher-uncertainty-smoke multiteacher-uncertainty-prior-smoke multiteacher-uncertainty-disagreement-smoke multiteacher-uncertainty-student-smoke multiteacher-uncertainty-student-prior-smoke multiteacher-uncertainty-composite-smoke multiteacher-learned-gate-smoke multiteacher-learned-linear-gate-smoke multiteacher-learned-prior-residual-smoke multiteacher-supervised-prior-residual-smoke multiteacher-supervised-hard-top1-smoke multiteacher-supervised-hard-top2-smoke distill-regression distill-regression-summary distill-lambda-sweep distill-lambda-summary distill-lambda-analysis distill-adaptive-smoke distill-adaptive-regression distill-adaptive-summary

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

gate-target-diagnostics:
	$(PYTHON) analyze_gate_targets.py

multiteacher-uniform-smoke:
	$(PYTHON) train.py --datasets caco2_wang --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 --seeds 42 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --multiteacher-strategy uniform --lambda-distill 1.0 --results-root results_multiteacher_uniform_smoke

multiteacher-validation-smoke:
	$(PYTHON) train.py --datasets caco2_wang --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 --seeds 42 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --multiteacher-strategy validation_weighted --lambda-distill 1.0 --results-root results_multiteacher_validation_smoke

multiteacher-top1-smoke:
	$(PYTHON) train.py --datasets caco2_wang --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 --seeds 42 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --multiteacher-strategy top1_validation --lambda-distill 1.0 --results-root results_multiteacher_top1_smoke

multiteacher-uncertainty-smoke:
	$(PYTHON) train.py --datasets caco2_wang --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 --seeds 42 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --multiteacher-strategy uncertainty_only --lambda-distill 1.0 --results-root results_multiteacher_uncertainty_smoke

multiteacher-uncertainty-prior-smoke:
	$(PYTHON) train.py --datasets caco2_wang --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 --seeds 42 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --multiteacher-strategy uncertainty_validation_prior --lambda-distill 1.0 --results-root results_multiteacher_uncertainty_prior_smoke

multiteacher-uncertainty-disagreement-smoke:
	$(PYTHON) train.py --datasets caco2_wang --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 --seeds 42 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --multiteacher-strategy uncertainty_teacher_disagreement --lambda-distill 1.0 --results-root results_multiteacher_uncertainty_disagreement_smoke

multiteacher-uncertainty-student-smoke:
	$(PYTHON) train.py --datasets caco2_wang --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 --seeds 42 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --multiteacher-strategy uncertainty_teacher_student --lambda-distill 1.0 --results-root results_multiteacher_uncertainty_student_smoke

multiteacher-uncertainty-student-prior-smoke:
	$(PYTHON) train.py --datasets caco2_wang --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 --seeds 42 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --multiteacher-strategy uncertainty_student_prior --lambda-distill 1.0 --results-root results_multiteacher_uncertainty_student_prior_smoke

multiteacher-uncertainty-composite-smoke:
	$(PYTHON) train.py --datasets caco2_wang --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 --seeds 42 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --multiteacher-strategy uncertainty_composite --lambda-distill 1.0 --results-root results_multiteacher_uncertainty_composite_smoke

multiteacher-learned-gate-smoke:
	$(PYTHON) train.py --datasets caco2_wang --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 --seeds 42 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --multiteacher-strategy learned_reliability_gate --lambda-distill 1.0 --results-root results_multiteacher_learned_gate_smoke

multiteacher-learned-linear-gate-smoke:
	$(PYTHON) train.py --datasets caco2_wang --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 --seeds 42 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --multiteacher-strategy learned_linear_gate --lambda-distill 1.0 --results-root results_multiteacher_learned_linear_gate_smoke

multiteacher-learned-prior-residual-smoke:
	$(PYTHON) train.py --datasets caco2_wang --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 --seeds 42 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --multiteacher-strategy learned_prior_residual_gate --lambda-distill 1.0 --results-root results_multiteacher_learned_prior_residual_smoke

multiteacher-supervised-prior-residual-smoke:
	$(PYTHON) train.py --datasets caco2_wang --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 --seeds 42 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --multiteacher-strategy learned_prior_residual_gate --lambda-distill 1.0 --lambda-gate-supervision 0.5 --results-root results_multiteacher_supervised_prior_residual_smoke

multiteacher-supervised-hard-top1-smoke:
	$(PYTHON) train.py --datasets caco2_wang --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 --seeds 42 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --multiteacher-strategy supervised_hard_top1_gate --lambda-distill 1.0 --lambda-gate-supervision 0.5 --results-root results_multiteacher_supervised_hard_top1_smoke

multiteacher-supervised-hard-top2-smoke:
	$(PYTHON) train.py --datasets caco2_wang --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 --seeds 42 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --multiteacher-strategy supervised_hard_top2_gate --lambda-distill 1.0 --lambda-gate-supervision 0.5 --results-root results_multiteacher_supervised_hard_top2_smoke

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
