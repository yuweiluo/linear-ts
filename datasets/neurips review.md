Long?

following standard approach covert classification to bandit problem,  regret and be thought of as a fraction of 



 pior literture .. classficaton , we measuee ... 

PYTHONPATH=src python -m emp1 -sim 5 -k 0 -para_min 200 -para_max 200 -para_gap 20 -pm 0 -nsd 0.2 -n 1 -mode "d" -gamma 0.1 -psd 0.01 -radius 5 -s 1 -alpha 0.0 -dataset 'mfeat-zernike'





PYTHONPATH=src python -m emp1 -sim 5 -k 0 -para_min 1000 -para_max 1000 -para_gap 20 -pm 0 -nsd 0.2 -n 1 -mode "d" -gamma 0.1 -psd 0.01 -radius 5 -s 1 -alpha 0.0 -dataset 'artificial-characters'





PYTHONPATH=src python -m emp1 -sim 5 -k 0 -para_min 230 -para_max 230 -para_gap 20 -pm 0 -nsd 0.2 -n 50 -mode "d" -gamma 0.1 -psd 0.01 -radius 5 -s 1 -alpha 0.0 -dataset 'segment'









PYTHONPATH=src python -m emp1 -sim 5 -k 0 -para_gap 20 -pm 0 -nsd 0.2 -n 1 -mode "d" -gamma 0.1 -psd 0.01 -radius 5 -s 1 -alpha 0.0 -dataset 'gesture' -para_min 980 -para_max 980



PYTHONPATH=src python -m emp1 -sim 5 -k 0 -para_gap 20 -pm 0 -nsd 0.2 -n 1 -mode "d" -gamma 0.1 -psd 0.01 -radius 5 -s 1 -alpha 0.0 -dataset 'JapaneseVowels' -para_min 150 -para_max 150



PYTHONPATH=src python -m emp1 -sim 5 -k 0 -para_gap 20 -pm 0 -nsd 0.2 -n 20 -mode "d" -gamma 0.1 -psd 0.01 -radius 5 -s 1 -alpha 0.0 -dataset 'JapaneseVowels' -para_min 200 -para_max 200





cardiotocography 2000 10.  20+ 50

segment 2300 7 20+ 50

mfeat-zernike 2000 10 20+?50

JapaneseVowels 2000 9     20 + 50 







We thank reviewer RqoU for the feedback and constructive comments. Below we summarize the comments and answer them respectively. We hope that our answers help address the reviewer’s concerns.

### The role of Section 5

At the end of Section 4, we have unified OFUL, LinTS, and Greedy in the same framework to derive frequentist regret bounds, as is presented in Theorem 1. However, as is stated in Remark 2 we still lack one component to make Theorem 1 practical, namely evaluating the regret ratio $\mu_t$, because it involves the unknown true parameter $\theta^\star$. 

The main purpose of Section 5 is to establish an upper bound of $\mu_t$ without knowledge of $\theta^\star$, so that we could apply Theorem 1 to calculate a **data-driven** frequentist regret bound, and to use it to improve the standard algorithms (i.e., we proposed LinTS-MR and Greedy-MR in Section 5.2). 

### Dropping $\tau_t $ in Section 5 

We drop $\tau_t$ in Section 5 just for clarification, because $\tau_t = 0$ corresponds to LinTS and Greedy which is well-known. In fact, we could relax this condition to any COFUL algorithms with bounded $\{\tau_t\}$ sequence, and the results in Section 5 are still valid because a bounded range of  $\{\alpha_t\}$ turns into a bounded range of $\{\tau_t\}$, and hence we could plug in the upper bound into Theorem 1. 

We would like to further remark that assuming bounded $\{\tau_t\}$ is reasonable.  Actually, it tunes the degree of optimism to consider in the optimistic reward, namely

$$
\tilde{x}_t=\underset{x \in \mathcal{X}_t}{\operatorname{argmax}}\left\langle x, \tilde{\theta}_t\right\rangle+\tau_t\|x\|{V_t^{-1}} \beta_t^{R L S}.
$$
When $\{\tau_t\}$ grows unbounded, it is like a policy that always chooses actions with the maximal uncertainty in reward. This is a trivially poor policy as it completely ignores the mean of the estimated reward. 

### Euclidean ball as the action space

We consider the Euclidean ball as a special case only for the continuous action space scenario. The same setting is also considered in Abeille etal. [2017], which established the only known (but suboptimal) frequentist result of LinTS. The general continuous action space scenario is challenging since computing the optimal action might be NP-hard for OFUL, which involves solving a bilinear program subject to the constraint induced by the action space. 

But, we also address discrete action sets and provided the corresponding results in Appendix A, while Abeille etal. [2017] didn’t discuss the discrete action set setting at all. 

### Bounds on the sensitivity ratio

We thank the reviewer for commenting on Theorem 2’s presentation. We will provide an explanation here and will update the paper accordingly. The presented $\Phi_t$ and $\Psi_t$ serve as the numerator and denominator of an upper bound $\widehat{\alpha}_t$ for $\alpha_t$, namely we have $\alpha_t \leq \widehat{\alpha}_t:=\Phi_t / \Psi_t.$ In fact, this bound holds with high probability for all time before $t$, as we presented In Theorem 2. 

### Explanations

Example 1 is just a toy example to illustrate a possible setting that leads to bounded $\{\alpha_t\}$. For Example 2, we have added a simulation in the one-page pdf response to validate that the mentioned scenario is not vacant empirically. 

### Source of the advantage

In summary, the advantage of TS-MR and Greedy-MR comes from the upper bounds in Section 5.2, and the ability they can dynamically switch between TS/Greedy and OFUL, depending on whether the current instance fails TS/Greedy or not. However, we would like to clarify that this dynamic switching is possible because Theorem 1 and 2 together provide a practical way to evaluate the performance of COFUL algorithms on the current instance. 

### Generalization beyond the unit sphere

As discussed above, general continuous space is challenging, and we focus on the unit sphere to illustrate our geometrical approach. Nevertheless, as is presented in Appendix D, our method works for discrete action sets that are common in real-world problems. 









We thank reviewer RqoU for the feedback and constructive comments. Below we summarize the comments and answer them respectively. We hope that our answers help address the reviewer’s concerns.

### The role of Section 5

At the end of Section 4, we have unified OFUL, LinTS, and Greedy in the same framework to derive frequentist regret bounds, as is presented in Theorem 1. However, as is stated in Remark 2 we still lack one component to make Theorem 1 practical, namely evaluating the regret ratio $\mu_t$, because it involves the unknown true parameter $\theta^\star$. 

The main purpose of Section 5 is to establish an upper bound of $\mu_t$ without knowledge of $\theta^\star$, so that we could apply Theorem 1 to calculate a **data-driven** frequentist regret bound, and to use it to improve the standard algorithms (i.e., we proposed LinTS-MR and Greedy-MR in Section 5.2). 

### Dropping $\tau_t $ in Section 5 

We drop $\tau_t$ in Section 5 just for clarification, because $\tau_t = 0$ corresponds to LinTS and Greedy which is well-known. In fact, we could relax this condition to any COFUL algorithms with bounded $\{\tau_t\}$ sequence, and the results in Section 5 are still valid because a bounded range of  $\{\alpha_t\}$ turns into a bounded range of $\{\tau_t\}$, and hence we could plug in the upper bound into Theorem 1. 

We would like to further remark that assuming bounded $\{\tau_t\}$ is reasonable.  Actually, it tunes the degree of optimism to consider in the optimistic reward, namely

$$
\tilde{x}_t=\underset{x \in \mathcal{X}_t}{\operatorname{argmax}}\left\langle x, \tilde{\theta}_t\right\rangle+\tau_t\|x\|{V_t^{-1}} \beta_t^{R L S}.
$$
When $\{\tau_t\}$ grows unbounded, it is like a policy that always chooses actions with the maximal uncertainty in reward. This is a trivially poor policy as it completely ignores the mean of the estimated reward. 

### Euclidean ball as the action space

We consider the Euclidean ball as a special case only for the continuous action space scenario. The same setting is also considered in Abeille etal. [2017], which established the only known (but suboptimal) frequentist result of LinTS. The general continuous action space scenario is challenging since computing the optimal action might be NP-hard for OFUL, which involves solving a bilinear program subject to the constraint induced by the action space. Please see, e.g., Page 1228 of 

- Daniel Russo and Benjamin Van Roy. Learning to optimize via posterior sampling. *Mathematics of Operations Research*, 39(4):1221–1243, 2014. doi: 10.1287/moor.2014.0650.

But, we also address discrete action sets and provided the corresponding results in Appendix A, while Abeille etal. [2017] didn’t discuss the discrete action set setting at all. 

### Bounds on the sensitivity ratio

We thank the reviewer for commenting on Theorem 2’s presentation. We will provide an explanation here and will update the paper accordingly. The presented $\Phi_t$ and $\Psi_t$ serve as the numerator and denominator of an upper bound $\widehat{\alpha}_t$ for $\alpha_t$, namely we have $\alpha_t \leq \widehat{\alpha}_t:=\Phi_t / \Psi_t.$ In fact, this bound holds with high probability for all time before $t$, as we presented In Theorem 2. 

### Explanations

Example 1 is just a toy example to illustrate a possible setting that leads to bounded $\{\alpha_t\}$. For Example 2, we have added a simulation in the global pdf response to validate that the mentioned scenario is not vacant empirically. 

### Source of the advantage

In summary, the advantage of TS-MR and Greedy-MR comes from the upper bounds in Section 5.2, and the ability they can dynamically switch between TS/Greedy and OFUL, depending on whether the current instance fails TS/Greedy or not. However, we would like to clarify that this dynamic switching is possible because Theorem 1 and 2 together provide a practical way to evaluate the performance of COFUL algorithms on the current instance. 

### Generalization beyond the unit sphere

As discussed above, general continuous space is challenging, and we focus on the unit sphere to illustrate our geometrical approach. Nevertheless, as is presented in Appendix D, our method works for discrete action sets that are common in real-world problems. 



a(1-x) + 1+x = (1+a) +(1-a)x < 0 

a>1

x>(1+a)/(a - 1)

$$ \langle x_t^{\star}, \theta^{\star}\rangle-\langle\tilde{x}_t, \theta^{\star}\rangle \leq2\|\tilde{x}_t\|_{V_t^{-1}} \beta_t^{R L S}(\delta^{\prime}), $$



$ \left\langle x_t^{\star}, \theta^{\star}\right\rangle-\left\langle\tilde{x}_t, \theta^{\star}\right\rangle \leq 2\left\|\tilde{x}_t\right\|_{V_t^{-1}} \beta_t^{R L S}\left(\delta^{\prime}\right)$

a(x-1) - x -1



a+1 - (a-1)x





We thank you for the valuable feedback. For the first concern regarding the norm of the actions, we would like to clarify that the frequentist bounds in Theorem 1 and Corollary 1 hold without this condition. They are general properties of COFUL algorithms. Instead, the norm condition is used in the proof of Theorem 2, when we bound the ratios $\\{\alpha_t\\}$ and $\\{\mu_t\\}$ in a data-driven manner for continuous action sets. Nevertheless, in Appendix A we provided methods to bound the ratios when the action sets are discrete, covering the majority of real-world problems. 

But we agree for continuous action sets, we still rely on the unit sphere/ball action set assumption. Nevertheless, due to the linearity or the reward, one can always write the actions as $x = \Vert x\Vert \cdot \frac{x}{\Vert x\Vert} $, and apply Theorem 2 to the normalized actions $ \frac{x}{\Vert x\Vert}$ . This way, the action set could be any set whose actions cover all directions, and have norms bounded from below and above, say $b\leq \Vert x \Vert \leq B$ for all $x \in \mathcal{X}$.  Then the new upper bound for $\alpha_t$ becomes $\hat \alpha_t \coloneqq  \frac{B \Phi_t}{b \Psi_t}  $.  We will add the discussion to the paper. 

Regarding the concerns of Case 2 and Proposition 4, 