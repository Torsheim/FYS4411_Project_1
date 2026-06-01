# Formula summary

The trial wave function is

```math
\Psi_T(R) = \prod_i \exp[-\alpha(x_i^2+y_i^2+\beta z_i^2)]
            \prod_{i<j} f(r_{ij}),
```

with

```math
f(r) = \begin{cases}
0, & r \le a,\\
1-a/r, & r>a.
\end{cases}
```

For the ideal spherical oscillator in `d` dimensions,

```math
E_L = \alpha N d + \left(\frac12 - 2\alpha^2\right)\sum_i r_i^2.
```

At the optimum `alpha=1/2`, this becomes the exact constant result

```math
E_0 = \frac{Nd}{2}.
```

The drift force is

```math
F_i = 2 \nabla_i \ln \Psi_T.
```

For the Gaussian part only,

```math
F_i = -4\alpha r_i
```

in the spherical case. In the 3D anisotropic case the z component is
`F_z = -4 alpha beta z`.

For the Jastrow factor, write `u(r)=log(1-a/r)`. Then

```math
u'(r) = \frac{a}{r(r-a)},
```

and

```math
u''(r) = -\frac{a(2r-a)}{r^2(r-a)^2}.
```

The full local energy is computed using

```math
E_L = -\frac12 \sum_i \frac{\nabla_i^2 \Psi_T}{\Psi_T}
      + \frac12\sum_i(x_i^2+y_i^2+\gamma^2z_i^2).
```
