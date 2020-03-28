(a) $y_w=1$ if and only if $w=o$，so the other terms that $w$ is not the outside word would be zero.

(b) 
$$
\begin{array}{}
&\mathbf{J}_{naive-softmax}(\bold{v}_c,o,\mathbf{U})\\
=&-\log P(O=o|C=c)\\
=&-\log\frac{\exp(\bold{u}_o^T\bold{v}_c)}{\sum_{w\in\text{Vocab}}\exp(\bold{u}_w^T\bold{v}_c)}\\
=&-\log[{\exp(\bold{u}_o^T\bold{v}_c)}]+\log[\sum_{w\in\text{Vocab}}\exp(\bold{u}_w^T\bold{v}_c)]\\

&\frac{\partial \mathbf{J}_{naive-softmax}(\bold{v}_c,o,\mathbf{U})}{\partial\bold{v}_c}\\
=& -\frac{1}{\exp(\bold{u}_o^T\bold{v}_c)}·\exp(\bold{u}_o^T\bold{v}_c)·\bold{u}_o+\frac{1}{\sum_{w\in\text{Vocab}}\exp(\bold{u}_w^T\bold{v}_c)}·\sum_{w\in\text{Vocab}}\exp(\bold{u}_w^T\bold{v}_c)\bold{u}_w\\
=& -\bold{u}_o+\frac{\sum_{w\in\text{Vocab}}\exp(\bold{u}_w^T\bold{v}_c)\bold{u}_w}{\sum_{w\in\text{Vocab}}\exp(\bold{u}_w^T\bold{v}_c)}\\
=& -\bold{u}_o+\sum_{w\in\text{Vocab}}\frac{\exp(\bold{u}_w^T\bold{v}_c)}{\sum_{x\in\text{Vocab}}\exp(\bold{u}_x^T\bold{v}_c)}·\bold{u}_w\\
=& -\bold{u}_o+\sum_{w\in\text{Vocab}}P(w|c)·\bold{u}_w\\
=& -U^T(\bold{y}-\bold{\hat{y}})
\end{array}
$$

(c)
$$
\begin{array}{rl}
\frac{\partial \mathbf{J}_{naive-softmax}(\bold{v}_c,o,\mathbf{U})}{\partial\bold{u}_o}
=& -\frac{1}{\exp(\bold{u}_o^T\bold{v}_c)}·\exp(\bold{u}_o^T\bold{v}_c)·\bold{v}_c+\frac{1}{\sum_{w\in\text{Vocab}}\exp(\bold{u}_w^T\bold{v}_c)}·\exp(\bold{u}_o^T\bold{v}_c)\bold{v}_c\\
=& -\bold{v}_c+\frac{\exp(\bold{u}_o^T\bold{v}_c)\bold{v}_c}{\sum_{w\in\text{Vocab}}\exp(\bold{u}_w^T\bold{v}_c)}\\
=& -\bold{v}_c+P(o|c)·\bold{v}_c\\
=&-(1-P(o|c))·\bold{v}_c\\
=&(P(o|c)-1)·\bold{v}_c\\

\frac{\partial \mathbf{J}_{naive-softmax}(\bold{v}_c,o,\mathbf{U})}{\partial\bold{u}_w}
=& \frac{1}{\sum_{x\in\text{Vocab}}\exp(\bold{u}_x^T\bold{v}_c)}·\exp(\bold{u}_w^T\bold{v}_c)\bold{v}_c\\
=& \frac{\exp(\bold{u}_w^T\bold{v}_c)\bold{v}_c}{\sum_{x\in\text{Vocab}}\exp(\bold{u}_x^T\bold{v}_c)}\\
=& (P(w|c)-0)·\bold{v}_c\\
 
{\color{red} in\ summary}&\\
{\color{red} 
\frac{\partial \mathbf{J}_{naive-softmax}(\bold{v}_c,o,\mathbf{U})}{\partial\bold{U}}}&
{\color{red} 
=(\bold{\hat{y}}-\bold{y})\bold{v}_c}
\end{array}
$$
(d)
$$
\begin{array}{}
\frac{\partial \sigma(x)}{\partial x}&=\frac{\partial\frac{e^x}{e^x+1}}{\partial x}\\
&=\frac{e^x(e^x+1)-e^x·e^x}{(e^x+1)^2}\\
&=\frac{e^x}{(e^x+1)^2}\\
&=\sigma(x)\sigma(-x)\\
&=\sigma(x)(1-\sigma(x))
\end{array}
$$

(e)
$$
\begin{array}{rl}
\frac{\partial\mathbf{J}_{neg-sample}(\bold{v}_c,o,\mathbf{U})}{\partial\bold{v}_c}
=&\frac{\partial-\log(\sigma(\bold{u}_o^T\bold{v}_c))-\sum_{k=1}^{K}\log(\sigma(-\bold{u}_k^T\bold{v}_c))}{\partial\bold{v}_c}\\
=&-\frac{\sigma(\bold{u}_o^T\bold{v}_c)\sigma(-\bold{u}_o^T\bold{v}_c)}{\sigma(\bold{u}_o^T\bold{v}_c)}·\bold{u}_o-\sum_{k=1}^{K}\frac{\sigma(-\bold{u}_k^T\bold{v}_c)\sigma(\bold{u}_k^T\bold{v}_c)}{\sigma(-\bold{u}_k^T\bold{v}_c)}·(-\bold{u}_k)\\
=&\sigma(\bold{u}_o^T\bold{v}_c)·\bold{u}_o-\bold{u}_o+\sum_{k=1}^{K}\sigma(\bold{u}_k^T\bold{v}_c)·\bold{u}_k\\
=&[\sum_{k\in[1,K]\cup\{o\}}\sigma(\bold{u}_k^T\bold{v}_c)·\bold{u}_k]-\bold{u}_o\\

\frac{\partial\mathbf{J}_{neg-sample}(\bold{v}_c,o,\mathbf{U})}{\partial\bold{u}_o}
=&\frac{\partial-\log(\sigma(\bold{u}_o^T\bold{v}_c))}{\partial\bold{u}_o}\\
=&-\frac{\sigma(\bold{u}_o^T\bold{v}_c)\sigma(-\bold{u}_o^T\bold{v}_c)}{\sigma(\bold{u}_o^T\bold{v}_c)}·\bold{v}_c\\
=&-\sigma(-\bold{u}_o^T\bold{v}_c)\bold{v}_c\\

\frac{\partial\mathbf{J}_{neg-sample}(\bold{v}_c,o,\mathbf{U})}{\partial\bold{u}_k}
=&-\frac{\sum_{k=1}^{K}\log(\sigma(-\bold{u}_k^T\bold{v}_c))}{\partial\bold{u}_k}\\
=&\frac{\sigma(-\bold{u}_k^T\bold{v}_c)\sigma(\bold{u}_k^T\bold{v}_c)}{\sigma(-\bold{u}_k^T\bold{v}_c)}·\bold{v}_c\\
=&\sigma(\bold{u}_k^T\bold{v}_c)·\bold{v}_c
\end{array}
$$
(f)

(i) $\frac{\partial\mathbf{J}_{skip-gram}(\bold{v}_c,w_{t-m},...,w_{t+m},\mathbf{U})}{\partial\bold{U}}=\frac{\partial\sum_{-m\leq j\leq m\\j\neq 0}\mathbf{J}(\bold{v}_c,w_{t+j},\mathbf{U})}{\partial\bold{U}}=\sum_{-m\leq j\leq m\\j\neq 0}\frac{\partial\mathbf{J}(\bold{v}_c,w_{t+j},\mathbf{U})}{\partial U}$

(ii) $\frac{\partial\mathbf{J}_{skip-gram}(\bold{v}_c,w_{t-m},...,w_{t+m},\mathbf{U})}{\partial\bold{v}_c}=\frac{\partial\sum_{-m\leq j\leq m\\j\neq 0}\mathbf{J}(\bold{v}_c,w_{t+j},\mathbf{U})}{\partial\bold{v}_c}=\sum_{-m\leq j\leq m\\j\neq 0}\frac{\partial\mathbf{J}(\bold{v}_c,w_{t+j},\mathbf{U})}{\partial\bold{v}_c}$

(iii) $\frac{\partial\mathbf{J}_{skip-gram}(\bold{v}_c,w_{t-m},...,w_{t+m},\mathbf{U})}{\partial\bold{v}_w}=0$