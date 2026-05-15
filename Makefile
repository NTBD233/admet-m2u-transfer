SHELL := /bin/bash
PYTHON ?= python

.PHONY: setup prepare features smoke train train-low-resource ml-baselines lambda-ablation analysis paper-tables summary teacher-predictions-regression teacher-predictions-multiteacher teacher-selector-smoke teacher-selector-regression teacher-reliability gate-target-diagnostics pretrained-selector-summary multiteacher-uniform-smoke multiteacher-validation-smoke multiteacher-top1-smoke multiteacher-uncertainty-smoke multiteacher-uncertainty-prior-smoke multiteacher-uncertainty-disagreement-smoke multiteacher-uncertainty-student-smoke multiteacher-uncertainty-student-prior-smoke multiteacher-uncertainty-composite-smoke multiteacher-learned-gate-smoke multiteacher-learned-linear-gate-smoke multiteacher-learned-prior-residual-smoke multiteacher-supervised-prior-residual-smoke multiteacher-supervised-hard-top1-smoke multiteacher-supervised-hard-top2-smoke pretrained-selector-top1-smoke pretrained-selector-top1-reweight-smoke pretrained-selector-top1-reweight-ppbr-smoke pretrained-selector-top1-disagreement-reweight-smoke pretrained-selector-top1-disagreement-reweight-ppbr-smoke pretrained-selector-top1-auto-reweight-smoke pretrained-selector-top1-auto-reweight-ppbr-smoke pretrained-selector-top1-auto-reweight-partial-regression pretrained-selector-confidence-fallback-top1-smoke pretrained-selector-validation-blend-top1-smoke pretrained-selector-top2-smoke selector-filtered-uniform-smoke selector-filtered-validation-smoke pretrained-selector-top1-regression pretrained-selector-top2-regression selector-filtered-validation-regression distill-regression distill-regression-summary distill-lambda-sweep distill-lambda-summary distill-lambda-analysis distill-adaptive-smoke distill-adaptive-regression distill-adaptive-summary

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

teacher-selector-smoke:
	$(PYTHON) train_teacher_selector.py --datasets caco2_wang --teachers ECFP4_RF Desc_RF ECFP4_Desc_RF --train-ratio-tags 10 --seeds 42 --selector-models rf logistic --output-root data/selector_predictions

teacher-selector-regression:
	$(PYTHON) train_teacher_selector.py --datasets caco2_wang lipophilicity_astrazeneca solubility_aqsoldb vdss_lombardo ppbr_az --teachers ECFP4_RF Desc_RF ECFP4_Desc_RF --train-ratio-tags 10 20 50 --seeds 42 123 3407 --selector-models rf --output-root data/selector_predictions --skip-existing

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

pretrained-selector-top1-smoke:
	$(PYTHON) train.py --datasets caco2_wang --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 --seeds 42 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --selector-root data/selector_predictions --selector-model-name rf_crossfit_train_pseudo_oracle --multiteacher-strategy pretrained_selector_top1 --lambda-distill 1.0 --results-root results_pretrained_selector_top1_smoke

pretrained-selector-top1-reweight-smoke:
	$(PYTHON) train.py --datasets caco2_wang --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 --seeds 42 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --selector-root data/selector_predictions --selector-model-name rf_crossfit_train_pseudo_oracle --multiteacher-strategy pretrained_selector_top1 --selector-distill-reweight --lambda-distill 1.0 --results-root results_pretrained_selector_top1_reweight_smoke

pretrained-selector-top1-reweight-ppbr-smoke:
	$(PYTHON) train.py --datasets ppbr_az --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 20 --seeds 42 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --selector-root data/selector_predictions --selector-model-name rf_crossfit_train_pseudo_oracle --multiteacher-strategy pretrained_selector_top1 --selector-distill-reweight --lambda-distill 1.0 --results-root results_pretrained_selector_top1_reweight_ppbr_smoke

pretrained-selector-top1-disagreement-reweight-smoke:
	$(PYTHON) train.py --datasets caco2_wang --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 --seeds 42 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --selector-root data/selector_predictions --selector-model-name rf_crossfit_train_pseudo_oracle --multiteacher-strategy pretrained_selector_top1 --selector-distill-reweight --selector-distill-reweight-mode disagreement_confidence --lambda-distill 1.0 --results-root results_pretrained_selector_top1_disagreement_reweight_smoke

pretrained-selector-top1-disagreement-reweight-ppbr-smoke:
	$(PYTHON) train.py --datasets ppbr_az --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 20 --seeds 42 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --selector-root data/selector_predictions --selector-model-name rf_crossfit_train_pseudo_oracle --multiteacher-strategy pretrained_selector_top1 --selector-distill-reweight --selector-distill-reweight-mode disagreement_confidence --lambda-distill 1.0 --results-root results_pretrained_selector_top1_disagreement_reweight_ppbr_smoke

pretrained-selector-top1-auto-reweight-smoke:
	$(PYTHON) train.py --datasets caco2_wang --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 --seeds 42 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --selector-root data/selector_predictions --selector-model-name rf_crossfit_train_pseudo_oracle --multiteacher-strategy pretrained_selector_top1 --selector-distill-reweight --selector-distill-reweight-mode auto --lambda-distill 1.0 --results-root results_pretrained_selector_top1_auto_reweight_smoke

pretrained-selector-top1-auto-reweight-ppbr-smoke:
	$(PYTHON) train.py --datasets ppbr_az --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 20 --seeds 42 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --selector-root data/selector_predictions --selector-model-name rf_crossfit_train_pseudo_oracle --multiteacher-strategy pretrained_selector_top1 --selector-distill-reweight --selector-distill-reweight-mode auto --lambda-distill 1.0 --results-root results_pretrained_selector_top1_auto_reweight_ppbr_smoke

pretrained-selector-top1-auto-reweight-partial-regression:
	$(PYTHON) train.py --datasets caco2_wang ppbr_az --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 20 50 --seeds 42 123 3407 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --selector-root data/selector_predictions --selector-model-name rf_crossfit_train_pseudo_oracle --multiteacher-strategy pretrained_selector_top1 --selector-distill-reweight --selector-distill-reweight-mode auto --lambda-distill 1.0 --results-root results_pretrained_selector_top1_auto_reweight_partial_regression --skip-existing

pretrained-selector-confidence-fallback-top1-smoke:
	$(PYTHON) train.py --datasets caco2_wang --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 --seeds 42 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --selector-root data/selector_predictions --selector-model-name rf_crossfit_train_pseudo_oracle --multiteacher-strategy pretrained_selector_confidence_fallback_top1 --selector-confidence-threshold 0.6 --lambda-distill 1.0 --results-root results_pretrained_selector_confidence_fallback_top1_smoke

pretrained-selector-validation-blend-top1-smoke:
	$(PYTHON) train.py --datasets caco2_wang --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 --seeds 42 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --selector-root data/selector_predictions --selector-model-name rf_crossfit_train_pseudo_oracle --multiteacher-strategy pretrained_selector_validation_blend_top1 --lambda-distill 1.0 --results-root results_pretrained_selector_validation_blend_top1_smoke

pretrained-selector-top2-smoke:
	$(PYTHON) train.py --datasets caco2_wang --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 --seeds 42 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --selector-root data/selector_predictions --selector-model-name rf_crossfit_train_pseudo_oracle --multiteacher-strategy pretrained_selector_top2 --lambda-distill 1.0 --results-root results_pretrained_selector_top2_smoke

selector-filtered-uniform-smoke:
	$(PYTHON) train.py --datasets caco2_wang --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 --seeds 42 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --selector-root data/selector_predictions --selector-model-name rf_crossfit_train_pseudo_oracle --multiteacher-strategy selector_filtered_uniform --lambda-distill 1.0 --results-root results_selector_filtered_uniform_smoke

selector-filtered-validation-smoke:
	$(PYTHON) train.py --datasets caco2_wang --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 --seeds 42 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --selector-root data/selector_predictions --selector-model-name rf_crossfit_train_pseudo_oracle --multiteacher-strategy selector_filtered_validation_weighted --lambda-distill 1.0 --results-root results_selector_filtered_validation_smoke

pretrained-selector-top1-regression:
	$(PYTHON) train.py --datasets caco2_wang lipophilicity_astrazeneca solubility_aqsoldb vdss_lombardo ppbr_az --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 20 50 --seeds 42 123 3407 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --selector-root data/selector_predictions --selector-model-name rf_crossfit_train_pseudo_oracle --multiteacher-strategy pretrained_selector_top1 --lambda-distill 1.0 --results-root results_pretrained_selector_top1_regression --skip-existing

pretrained-selector-top2-regression:
	$(PYTHON) train.py --datasets caco2_wang lipophilicity_astrazeneca solubility_aqsoldb vdss_lombardo ppbr_az --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 20 50 --seeds 42 123 3407 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --selector-root data/selector_predictions --selector-model-name rf_crossfit_train_pseudo_oracle --multiteacher-strategy pretrained_selector_top2 --lambda-distill 1.0 --results-root results_pretrained_selector_top2_regression --skip-existing

selector-filtered-validation-regression:
	$(PYTHON) train.py --datasets caco2_wang lipophilicity_astrazeneca solubility_aqsoldb vdss_lombardo ppbr_az --models ECFP4_MLP_DescAdapterFusion --train-ratio-tags 10 20 50 --seeds 42 123 3407 --teacher-root data/teacher_predictions --teacher-models ECFP4_RF Desc_RF ECFP4_Desc_RF --selector-root data/selector_predictions --selector-model-name rf_crossfit_train_pseudo_oracle --multiteacher-strategy selector_filtered_validation_weighted --lambda-distill 1.0 --results-root results_selector_filtered_validation_regression --skip-existing

pretrained-selector-summary:
	$(PYTHON) compare_pretrained_selector.py --results base=results selector_top1=results_pretrained_selector_top1_regression selector_top2=results_pretrained_selector_top2_regression selector_filtered=results_selector_filtered_validation_regression --output-root results_pretrained_selector_analysis

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
