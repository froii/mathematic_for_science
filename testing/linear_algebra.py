import numpy as np

print("=" * 60)
print("ОСНОВИ РОБОТИ З МАТРИЦЯМИ")
print("=" * 60)

# Створюємо матрицю розміром 4×3 з довільними цілими числами
# 4 — це кількість рядків (i, або вісь 0, часто позначають як вертикальну вісь)
# 3 — це кількість стовпців (j, або вісь 1, часто позначають як горизонтальну вісь)
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

print("\nМатриця:")
print(matrix)
print()

# 1. Виводимо розмірність матриці
print("Розмірність матриці:", matrix.shape)
print()

# 2. Виводимо елемент на позиції (2, 3)
# Зверніть увагу: індексація в NumPy починається з 0
# Позиція (2, 3) означає 2-й рядок, 3-й стовпець (індекси 1, 2)
print("Елемент на позиції (2, 3):", matrix[1, 2])
print()

# 3. Виводимо другий рядок
print("Другий рядок:", matrix[1])
print()

# 4. Виводимо третій стовпець
print("Третій стовпець:", matrix[:, 2])

print("\n" + "=" * 60)
print("ТРАНСПОНУВАННЯ МАТРИЦІ")
print("=" * 60)

# Створюємо матрицю A
A = np.array([[1, 2, 3], [4, 5, 6]])
print("\nМатриця A:")
print(A)
print(f"Розмір A: {A.shape}")

# Обчислюємо транспоновану матрицю A^T
A_T = A.T
print("\nТранспонована матриця A^T:")
print(A_T)
print(f"Розмір A^T: {A_T.shape}")

# Перевіряємо властивість (A^T)^T = A
A_T_T = A_T.T
print("\nПодвійне транспонування (A^T)^T:")
print(A_T_T)

# Програмна перевірка рівності
is_equal = np.array_equal(A_T_T, A)
print(f"\nПеревірка властивості (A^T)^T = A: {is_equal}")

print("\n" + "=" * 60)
print("ОПЕРАЦІЇ З МАТРИЦЯМИ")
print("=" * 60)

# Створюємо дві матриці 3×3 з випадковими числами від 1 до 10
np.random.seed(42)  # для відтворюваності результатів
A = np.random.randint(1, 11, (3, 3))
B = np.random.randint(1, 11, (3, 3))

print("\nМатриця A:")
print(A)
print("\nМатриця B:")
print(B)

# Додавання матриць
print("\n1. Додавання матриць (A + B):")
print(A + B)

# Віднімання матриць
print("\n2. Віднімання матриць (A - B):")
print(A - B)

# Множення на скаляр
print("\n3. Множення матриці A на скаляр 3:")
print(3 * A)

# Лінійна комбінація
print("\n4. Лінійна комбінація 2A + 3B:")
print(2 * A + 3 * B)

print("\n" + "=" * 60)
print("ДОДАТКОВІ ОПЕРАЦІЇ")
print("=" * 60)

# Поелементне множення (Hadamard product)
print("\n5. Поелементне множення A ⊙ B:")
print(A * B)

# Матричне множення
print("\n6. Матричне множення A @ B:")
print(A @ B)
# Альтернатива: np.dot(A, B)

# Визначник матриці
print("\n7. Визначник матриці A:")
det_A = np.linalg.det(A)
print(f"det(A) = {det_A:.2f}")

# Слід матриці (trace - сума діагональних елементів)
print("\n8. Слід матриці A (сума діагональних елементів):")
trace_A = np.trace(A)
print(f"trace(A) = {trace_A}")

# Обернена матриця (якщо визначник != 0)
if det_A != 0:
    print("\n9. Обернена матриця A^(-1):")
    A_inv = np.linalg.inv(A)
    print(A_inv)

    # Перевірка: A @ A^(-1) = I (одинична матриця)
    print("\n10. Перевірка: A @ A^(-1) ≈ I (одинична матриця):")
    identity_check = A @ A_inv
    print(np.round(identity_check, decimals=10))
else:
    print("\n9. Матриця A вироджена (det = 0), обернена матриця не існує")

# Норма матриці
print("\n11. Норма Фробеніуса матриці A:")
frobenius_norm = np.linalg.norm(A, "fro")
print(f"||A||_F = {frobenius_norm:.2f}")

# Власні значення та власні вектори
print("\n12. Власні значення та власні вектори матриці A:")
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Власні значення:")
print(eigenvalues)
print("Власні вектори:")
print(eigenvectors)

# Ранг матриці
print("\n13. Ранг матриці A:")
rank_A = np.linalg.matrix_rank(A)
print(f"rank(A) = {rank_A}")

print("\n" + "=" * 60)
print("СПЕЦІАЛЬНІ МАТРИЦІ")
print("=" * 60)

# Одинична матриця
print("\n14. Одинична матриця 3×3:")
identity = np.eye(3)
print(identity)

# Нульова матриця
print("\n15. Нульова матриця 2×4:")
zeros = np.zeros((2, 4))
print(zeros)

# Матриця з одиниць
print("\n16. Матриця з одиниць 3×2:")
ones = np.ones((3, 2))
print(ones)

# Діагональна матриця
print("\n17. Діагональна матриця:")
diag_matrix = np.diag([1, 2, 3, 4])
print(diag_matrix)

print("\n" + "=" * 60)
print("ЗАВЕРШЕНО")
print("=" * 60)
