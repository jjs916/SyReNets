Note anything
12.12.2022 - First meeting: Philipp, Fan, and Jaesug

Math1: $x\in\mathbb{R}^2$, $y=x_1^2+x_2$

Works well ( $\lambda_2=0.001,0.01,0.1,1$ )


Math2: $x\in\mathbb{R}^2$, $y=2.3x_1^2+2x_1x_2-0.5x_2$

Math3: $x\in\mathbb{R}^2$, $y=x_1^2+x_1x_2-2x_2^3$

Works well with {"depth": 4, "n_selection_heads": 2, "samples": 32000, "n_iteration": 20000}. For all $\lambda_2=0.001,0.01,0.1,1$ it identifies the formula, however, cannot eliminate all the unneccessary terms, meaning there are many coefficients between 0.1 and 0.01 which disturb the prediction. $\lambda_2=0.1$ works the best.


Math4: $x\in\mathbb{R}^3$, $y=x_1^2+x_1x_2^2-x_2^3$

Math5: $x\in\mathbb{R}^5$, $y=x_1^2+x_2+x_1x_2^2-x_2^3+2.1x_2x_5^2$

Math6: $x\in\mathbb{R}^2$, $y=\cos(x_1)^2+x_2$

Math7: $x\in\mathbb{R}^2$, $y=x_0^2 +  \sin(x_2) -2.3 \cos(0.1 x_2)$

Math8: $x\in\mathbb{R}^2$, $y=x_1^2+x_2 + 1$
