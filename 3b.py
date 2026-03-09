import numpy as np
import matplotlib.pyplot as plt

n = 100
x_verdi = np.linspace(-1, 1, n)
def f(x):
    return np.cos(np.pi*x)

def u(x):
    u_analytisk = -(1.0 / np.pi**2)*f(x) + x + 1.0 - (1.0 / np.pi**2)
    return u_analytisk

y_verdi = u(x_verdi)

def u_numerisk(n):
    x_num_verdi = np.linspace(-1, 1, n+1)
    h = x_num_verdi[1] - x_num_verdi[0]
    A = np.zeros((n-1, n-1))
    f = np.cos(np.pi*x_num_verdi)
    for i in range(n-1):
        A[i,i] = -2.0
        if i > 0:
            A[i,i-1] = 1.0
        if i < n-2:
            A[i,i+1] = 1.0
            
    b = (h**2) * f[1:-1]
    # b for randsbetingelsene: u(-1) = 0 og u(1) = 2
    b[0] -= 0.0
    b[-1] -= 2.0
    # løser Ax = b
    u_lin = np.linalg.solve(A, b)
    # hele løsningen
    u_num_los = np.concatenate(([0.0], u_lin, [2.0]))
    return x_num_verdi, u_num_los

# numerisk data
num_intervaller = 15
x_num, y_num = u_numerisk(num_intervaller)


plt.figure(figsize=(8, 5))
plt.plot(x_verdi, y_verdi, label="Analytisk løsning", color='blue')
plt.plot(x_num, y_num, 'r--', label=f"Numerisk løsning (n={num_intervaller})")
plt.scatter(x_num, y_num, color='red', s=20)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.show()
