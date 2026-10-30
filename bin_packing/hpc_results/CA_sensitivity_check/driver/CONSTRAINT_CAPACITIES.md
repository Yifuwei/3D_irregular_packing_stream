# Weight and radiation constraint capacities

Source: EXPERIMENT_PARAMETERS.json / calibration_preview. Based on five sequence seeds per dataset and bin shape.

U_limit = weight_capacity / (weight_factor * bin_capacity). U_limit is a maximum allowed filling ratio, not the measured U or U*. Values below are displayed as percentages; source precision is unchanged.

Runtime reads absolute weight_capacity into max_weight, with weight_factor=1 and rho=1. Eligibility checks sum(piece.volume)*weight_factor <= max_weight. Radiation values are absolute capacities.

| Dataset | Bin shape | U weak | U medium | U strong | Radiation weak | Radiation medium | Radiation strong |
|---|---|---:|---:|---:|---:|---:|---:|
| Merged1_normal | cube | 31.3246% | 23.4935% | 15.6623% | 2934.913272 | 2014.913272 | 1094.913272 |
| Merged1_normal | cylinder | 27.4001% | 20.5500% | 13.7000% | 3018.822364 | 2058.822364 | 1098.822364 |
| Merged2_normal | cube | 36.8937% | 27.6703% | 18.4469% | 3515.242265 | 2305.242265 | 1095.242265 |
| Merged2_normal | cylinder | 31.1509% | 23.3632% | 15.5755% | 4056.863283 | 2576.863283 | 1096.863283 |
| Merged3_normal | cube | 36.1889% | 27.1417% | 18.0945% | 3266.372446 | 2183.039113 | 1099.705779 |
| Merged3_normal | cylinder | 33.1493% | 24.8620% | 16.5747% | 3870.036901 | 2480.036901 | 1090.036901 |
| Merged4_normal | cube | 36.3186% | 27.2389% | 18.1593% | 3448.793104 | 2273.793104 | 1098.793104 |
| Merged4_normal | cylinder | 31.6170% | 23.7128% | 15.8085% | 3599.126253 | 2345.792919 | 1092.459586 |
| Merged5_normal | cube | 37.2367% | 27.9275% | 18.6183% | 3502.374932 | 2297.374932 | 1092.374932 |
| Merged5_normal | cylinder | 32.7146% | 24.5359% | 16.3573% | 3638.666969 | 2368.666969 | 1098.666969 |
| St_05_Example3_normal | cube | 37.6951% | 28.2713% | 18.8475% | 3125.734432 | 2105.734432 | 1085.734432 |
| St_05_Example3_normal | cylinder | 35.7996% | 26.8497% | 17.8998% | 3177.007190 | 2137.007190 | 1097.007190 |
| chess | cube | 27.2755% | 20.4566% | 13.6378% | 2715.747438 | 1905.747438 | 1095.747438 |
| chess | cylinder | 25.1761% | 18.8821% | 12.5881% | 2673.297940 | 1883.297940 | 1093.297940 |
| engine | cube | 27.2205% | 20.4153% | 13.6102% | 18019.007871 | 9559.007871 | 1099.007871 |
| engine | cylinder | 23.7096% | 17.7822% | 11.8548% | 18457.507045 | 9777.507045 | 1097.507045 |
| liu | cube | 21.9242% | 16.4432% | 10.9621% | 1096.326121 | 1096.326121 | 1096.326121 |
| liu | cylinder | 19.9691% | 14.9768% | 9.9846% | 1828.479400 | 1463.479400 | 1098.479400 |
| shapesnew | cube | 24.9186% | 18.6890% | 12.4593% | 1099.482503 | 1099.482503 | 1099.482503 |
| shapesnew | cylinder | 24.0471% | 18.0353% | 12.0236% | 1351.866783 | 1218.533450 | 1085.200116 |
| st04_Example2_normal | cube | 20.7209% | 15.5407% | 10.3605% | 3933.804481 | 2513.804481 | 1093.804481 |
| st04_Example2_normal | cylinder | 15.2004% | 11.4003% | 7.6002% | 4093.224380 | 2593.224380 | 1093.224380 |
| st04_Example3_normal | cube | 24.2591% | 18.1943% | 12.1295% | 5017.925467 | 3057.925467 | 1097.925467 |
| st04_Example3_normal | cylinder | 19.5405% | 14.6554% | 9.7703% | 5455.773666 | 3275.773666 | 1095.773666 |
| st04_Example4_normal | cube | 22.1941% | 16.6456% | 11.0971% | 4619.484005 | 2859.484005 | 1099.484005 |
| st04_Example4_normal | cylinder | 21.3006% | 15.9755% | 10.6503% | 6419.597177 | 3759.597177 | 1099.597177 |
| st04_Example5_normal | cube | 22.7626% | 17.0719% | 11.3813% | 4537.894407 | 2817.894407 | 1097.894407 |
| st04_Example5_normal | cylinder | 22.0735% | 16.5552% | 11.0368% | 7016.758203 | 4056.758203 | 1096.758203 |
| st05_e2 | cube | 33.2100% | 24.9075% | 16.6050% | 2453.708720 | 1773.708720 | 1093.708720 |
| st05_e2 | cylinder | 30.6207% | 22.9656% | 15.3104% | 2411.632562 | 1751.632562 | 1091.632562 |

Strong means a smaller allowed capacity. U limits include the largest-piece feasibility floor, so they need not be exactly 0.5/0.75/1 times the reference U*.
For cylinder bins the normalization volume is pi*(diameter/2)^2*height; for cube bins it is length*width*height.
If n* <= 5, all three radiation capacities coincide. Display rounding does not affect runtime inputs.
