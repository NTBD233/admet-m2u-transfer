| method | runs | mean_test_rmse | wins_base | delta_base | wins_fixed | delta_fixed | wins_top1 | delta_top1 | wins_selector | delta_selector |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base AdapterFusion | 45 | 5.1841 |  |  | 15/45 | 0.0221 | 11/45 | 0.1049 | 11/45 | 0.0953 |
| Fixed ECFP4+Desc RF | 45 | 5.1621 | 30/45 | -0.0221 |  |  | 19/45 | 0.0828 | 17/45 | 0.0732 |
| Uniform multi-teacher | 45 | 5.1403 | 29/45 | -0.0438 | 23/45 | -0.0217 | 24/45 | 0.0611 | 21/45 | 0.0514 |
| Validation-weighted | 45 | 5.0969 | 33/45 | -0.0872 | 24/45 | -0.0651 | 19/45 | 0.0177 | 19/45 | 0.0081 |
| Top-1 validation | 45 | 5.0792 | 34/45 | -0.1049 | 26/45 | -0.0828 |  |  | 20/45 | -0.0096 |
| Pretrained selector | 45 | 5.0889 | 34/45 | -0.0953 | 28/45 | -0.0732 | 25/45 | 0.0096 |  |  |
| Selector + high-resource decay | 45 | 5.0814 | 34/45 | -0.1028 | 32/45 | -0.0807 | 27/45 | 0.0021 | 9/45 | -0.0075 |
