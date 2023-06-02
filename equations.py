#!/usr/bin/env python
# coding: utf-8

# In[1]:


# загружаем зависимости
get_ipython().system('pip install sympy')
get_ipython().system('pip install numpy')


# In[2]:


# библиотека для символьных вычислений
from sympy import Rational, sqrt, I, symbols, simplify, Eq, latex
import numpy as np
from itertools import product


# In[3]:


# задаём константы

eps = -Rational(1, 2) + sqrt(3)/2*I

# t_{ij}^{(m)} <=> t[m][i][j]
t = np.array([
    [
        [eps, 0, 0],
        [0, 1, 0],
        [0, 0, eps**2],
    ],
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ],
    [
        [eps**2, 0, 0],
        [0, 1, 0],
        [0, 0, eps],
    ],
    [
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ],
    [
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0],
    ],
    [
        [0, eps, 1],
        [eps, 0, eps**2],
        [1, eps**2, 0],
    ],
    [
        [0, eps**2, 1],
        [eps**2, 0, eps],
        [1, eps, 0],
    ],
    [
        [0, -eps**2, 1],
        [eps**2, 0, -eps],
        [-1, eps, 0],
    ],
    [
        [0, -eps, 1],
        [eps, 0, -eps**2],
        [-1, eps**2, 0],
    ],
])

t[4] *= I / sqrt(3)
t[7] *= I / sqrt(3)
t[8] *= I / sqrt(3)

# h_{d,m}
h = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1],  # phi1
    [1, 1, 1, 1, -1, 1, 1, -1, -1],  # phi2
    [eps, 1, eps**2, 1, -1, eps**2, eps, -eps, -eps**2],  # phi3
    [eps**2, 1, eps, 1, -1, eps, eps**2, -eps**2, -eps],  # phi4
    [eps**2, 1, eps, 1, 1, eps, eps**2, eps**2, eps],  # phi5
    [eps, 1, eps**2, 1, 1, eps**2, eps, eps, eps**2],  # phi6
])


# In[4]:


# задаём перестановки

# psi: (0, 1, 2, 3, 4, 5, 6, 7, 8)
def psi_even(m):
    return m

# psi: (2, 1, 0, 3, 4, 6, 5, 8, 7)


def psi_odd(m):
    match m:
        case 0:
            return 2
        case 2:
            return 0
        case 5:
            return 6
        case 6:
            return 5
        case 7:
            return 8
        case 8:
            return 7
        case _:
            return m


psi = np.array([
    psi_even,
    psi_odd,
    psi_odd,
    psi_odd,
    psi_even,
    psi_even,
])


# In[5]:


# 54 переменные искомой системы
X = symbols("x:9")
Y = symbols("y:9")
Z = symbols("z:9")
A = symbols("a:9")
B = symbols("b:9")
C = symbols("c:9")


# In[6]:


# phi_d(x_m)
def phi(d, X, m):
    return h[d-1][m]*X[psi[d-1](m)]


# правая часть уравнения из исходной системы
def f(i, j, k, l, r, s):
    if (i == j and k == l and r == s) and not (i == j == k == l == r == s):
        return 1
    if (j == k and l == r and s == i) and not (i == j == k == l == r == s):
        return -1
    return 0


# In[7]:


# левая часть уравнения
def generate_left(m, p, q):
    left = 2*(X[m]*Y[p]*Z[q] + X[p]*Y[q]*Z[m] + X[q]*Y[m]*Z[p] +
              phi(2, X, m)*phi(2, Y, p)*phi(2, Z, q) +
              phi(2, X, p)*phi(2, Y, q)*phi(2, Z, m) +
              phi(2, X, q)*phi(2, Y, m)*phi(2, Z, p))
    for d in range(3, 7):
        left += phi(d, A, m)*phi(d, B, p)*phi(d, C, q) +            phi(d, A, p)*phi(d, B, q)*phi(d, C, m) +            phi(d, A, q)*phi(d, B, m)*phi(d, C, p)
    return simplify(left)


# правая часть уравнения
def generate_right(m, p, q):
    right = 0
    for i, j, k, l, r, s in product(range(3), repeat=6):
        right += t[m][i][j]*t[p][k][l]*t[q][r][s]*f(i, j, k, l, r, s)
    return simplify(right)


def generate_equation(m, p, q):
    left = generate_left(m, p, q)
    right = generate_right(m, p, q)
    return Eq(left, right)


equations = []
is_counted = np.zeros((9, 9, 9), bool)
with open("equations.tex", "w") as equations_file:
    for m, p, q in product(range(9), repeat=3):
        if is_counted[m][p][q]:
            continue

        # поскольку уравнения для шестёрок (m, p, q), (p, q, m), (q, m, p),
        # (phi_2(m), phi_2(p), phi_2(q)), (phi_2(p), phi_2(q), phi_2(m)), (phi_2(q), phi_2(m), phi_2(p))
        # оказываются идентичными
        is_counted[m][p][q] = True
        is_counted[p][q][m] = True
        is_counted[q][m][p] = True
        is_counted[psi[1](m)][psi[1](p)][psi[1](q)] = True
        is_counted[psi[1](p)][psi[1](q)][psi[1](m)] = True
        is_counted[psi[1](q)][psi[1](m)][psi[1](p)] = True

        equation = generate_equation(m, p, q)
        if equation == True:
            continue
        equations.append(equation)
        equations_file.write(f"\\[ {latex(equation)} \\]\n")


# In[8]:


# для уравнений в сжатом виде
XYZ = symbols("xyz:9:9:9")
ABC = symbols("abc:9:9:9")


# In[9]:


def get_index(m, p, q):
    return 81*m + 9*q + p


# phi_d(p_{mpq})
def phi(d, XYZ, m, p, q):
    return h[d-1][m]*h[d-1][p]*h[d-1][q]*XYZ[get_index(psi[d-1](m), psi[d-1](p), psi[d-1](q))]


# In[10]:


# левая часть уравнения
def generate_left(m, p, q):
    left = 2*(XYZ[get_index(m, p, q)] + phi(2, XYZ, m, p, q))
    for d in range(3, 7):
        left += phi(d, ABC, m, p, q)
    return simplify(left)


compressed_equations = []
is_counted = np.zeros((9, 9, 9), bool)
with open("compressed_equations.tex", "w") as equations_file:
    for m, p, q in product(range(9), repeat=3):
        if is_counted[m][p][q]:
            continue

        # поскольку уравнения для шестёрок (m, p, q), (p, q, m), (q, m, p),
        # (phi_2(m), phi_2(p), phi_2(q)), (phi_2(p), phi_2(q), phi_2(m)), (phi_2(q), phi_2(m), phi_2(p))
        # оказываются идентичными
        is_counted[m][p][q] = True
        is_counted[p][q][m] = True
        is_counted[q][m][p] = True
        is_counted[psi[1](m)][psi[1](p)][psi[1](q)] = True
        is_counted[psi[1](p)][psi[1](q)][psi[1](m)] = True
        is_counted[psi[1](q)][psi[1](m)][psi[1](p)] = True

        equation = generate_equation(m, p, q)
        if equation == True:
            continue
        compressed_equations.append(equation)
        equations_file.write(f"\\[ {latex(equation)} \\]\n")

