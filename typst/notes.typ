#import "/template/template.typ": *
#show: rules

#set math.mat(delim: "[")



data points : ${bold(P)_i}_(i=1)^N subset.eq RR^d$

$ bold(S)(t) = sum_(j=0)^n B_j^p (t) dot bold(C)_j $
- spline: $bold(S)(t) in RR^d$
- control points: ${bold(C)_j}_(j=0)^n subset.eq RR^d, quad n := m-p-1$
- parametrization: ${t_i}_(i=1)^N subset.eq RR$ such that $bold(S)(t_i) approx bold(P)_i$
- knots: ${u_h}_(h=0)^m subset.eq RR$ non-decreasing
- basis functions (degree $p$): $B_j^p (t)$ $->$ Cox--de Boor recursion formula:
$
  &B_j^0 (t) := cases(
    thin 1 thin \, quad u_j <= t <= u_(j+1) ,
    thin 0 thin \, quad "otherwise"
  )
\
  &B_j^p (t) := 
  vfrac(t - u_j, u_(j+p) - u_j) B_j^(p-1) (t) + 
  vfrac(u_(j+p+1) - t, u_(j+p+1) - u_(j+1)) B_(j+1)^(p-1) (t) 
  thick , quad j = 0, ..., underbrace(m-p-1, n)
$

#line(length: 100%, stroke: 0.5pt)


given knots $bold(U) = (u_0, ..., u_m)$ and parametrization $bold(T) = (t_1, ..., t_N)$, \
compute optimal control points $bold(C)$ by solving
$
  min sum_(i=1)^N norm( bold(P)_i - bold(S)(t_i) )_2^2 \
  = min norm( bold(P) - B bold(C) )_F^2
$
equivalent to the normal equation $wide (B^T B) bold(C) = B^T bold(P) wide$ where $B$ depends on $bold(U)$ and $bold(T)$
$
  B := mat(
    B_0^p (t_1), ..., B_n^p (t_1);
    dots.v, dots.down, dots.v;
    B_0^p (t_N), ..., B_n^p (t_N);
  ) in RR^(N times (n+1))
, wide
  bold(C) = vec(bold(C)_0, dots.v, bold(C)_n) in RR^((n+1) times d)
, wide
  bold(P) = vec(bold(P)_1, dots.v, bold(P)_N) in RR^(N times d)
$

#line(length: 100%, stroke: 0.5pt)
#pagebreak()


== de Boor algorithm

B-spline basis functions have _local support_: $B_j^p (t) = 0$ for $t in.not [u_j, u_(j+p+1))$.

Suppose $t in [u_r, u_(r+1))$. Then the only basis functions of degree $p$ which are not zero in $t$ are $B_(r-p)^p, ..., B_r^p$. Thus
$
  S(t) &= sum_(j = r-p)^r B_j^p (t) med C_j
\
  &= sum_(j = r-p)^r [
    vfrac(t - u_j, u_(j+p) - u_j) B_j^(p-1) (t) + 
    vfrac(u_(j+p+1) - t, u_(j+p+1) - u_(j+1)) B_(j+1)^(p-1) (t)
  ] C_j
\
  &= sum_(j = r-p+1)^r vfrac(t - u_j, u_(j+p) - u_j) B_j^(p-1) (t) med C_j
  + sum_(j = r-p)^(r-1) vfrac(u_(j+p+1) - t, u_(j+p+1) - u_(j+1)) B_(j+1)^(p-1) (t) med C_j
\
  &= sum_(j = r-p+1)^r vfrac(t - u_j, u_(j+p) - u_j) B_j^(p-1) (t) med C_j
  + sum_(j = r-p+1)^(r) vfrac(u_(j+p) - t, u_(j+p) - u_(j)) B_(j)^(p-1) (t) med C_(j-1)
\
  &= sum_(j = r-p+1)^r B_(j)^(p-1) (t) [vfrac(t - u_j, u_(j+p) - u_j) C_j + vfrac(u_(j+p) - t, u_(j+p) - u_(j)) C_(j-1)]
$
where we used that in the first sum the first term $B_(r-p)^(p-1)$ has support $[u_(r-p), u_r)$ \ and in the second sum the last term $B_(r+1)^(p-1)$ has support $[u_(r+1), u_(r+p+1))$.

If we define, for $thin j = r-p+1, ..., r$
$
  alpha_j := vfrac(t - u_j, u_(j+p) - u_j) thin , 
  wide "so that" quad 
  1-alpha_j = vfrac(u_(j+p) - t, u_(j+p) - u_(j))
\
  "and" quad tilde(C)_j = alpha_j C_j + (1-alpha_j) C_(j-1)
$
then we can rewrite
$
  S(t) = sum_(j = r-p+1)^r B_(j)^(p-1) (t) med tilde(C)_j
$
We can continue this by induction.

Let, for $thin k = 1, ..., p thin$ and $thin j = r-p+k, ..., r$
$
  &C_j^((0)) := C_j
\
  &C_j^((k)) := \(1-alpha_j^((k))) C_(j-1)^((k-1)) + alpha_j^((k)) C_j^((k-1))
\
  &wide alpha_j^((k)) := vfrac(t - u_j, u_(j+p+1-k) - u_j)
$
Then at every iteration $k$ we have
$
  S(t) = sum_(j = r-p+k)^r B_(j)^(p-k) (t) med C_j^((k))
$
and for $k = p$ we have $S(t) = C_r^((p))$.

#pagebreak()

It is more convenient to use a $0$-based index, so if we substitute $thin j = r-p+i thin$ \ we can define for $thin i = k, ..., p thin$
$
  &&D_i^((k)) &:= C_(r-p+i)^((k))
\
  &&&= \(1-alpha_(r-p+i)^((k))) C_((r-p+i)-1)^((k-1)) + alpha_(r-p+i)^((k)) C_(r-p+i)^((k-1))
\
  &&&= \(1-tilde(alpha)_(i)^((k))) D_(i-1)^((k-1)) + tilde(alpha)_(i)^((k)) D_i^((k-1))
\
  &&tilde(alpha)_(i)^((k)) &:= alpha_(r-p+i)^((k)) = vfrac(t - u_(r-p+i), u_(r+i+1-k) - u_(r-p+i))
$

If we iterate $i$ down, we can reuse memory: for $thin i = p, ..., k$
$
  D_i <- \(1-tilde(alpha)_(i)) D_(i-1) + tilde(alpha)_(i) D_i
\
  tilde(alpha)_(i) <- vfrac(t - u_(r-p+i), u_(r+i+1-k) - u_(r-p+i))
$
The final result is in $D_p$.