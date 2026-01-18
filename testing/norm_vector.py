import matplotlib.pyplot as plt
import numpy as np

print("=" * 70)
print("ЗАДАЧА 1: ОБЧИСЛЕННЯ РІЗНИХ НОРМ ВЕКТОРА")
print("=" * 70)

# Вектор v = [3, -4, 5]
v = np.array([3, -4, 5])
print(f"\nВектор v = {v}")

# 1. Евклідова норма (L2-норма) - найпоширеніша
euclidean_norm = np.linalg.norm(v, 2)
print(f"\n1. Евклідова норма (L2): ||v||₂ = {euclidean_norm:.4f}")
print(f"   Формула: √(3² + (-4)² + 5²) = √(9 + 16 + 25) = √50 = {euclidean_norm:.4f}")

# 2. Мангеттенська норма (L1-норма) - сума абсолютних значень
manhattan_norm = np.linalg.norm(v, 1)
print(f"\n2. Мангеттенська норма (L1): ||v||₁ = {manhattan_norm:.4f}")
print(f"   Формула: |3| + |-4| + |5| = 3 + 4 + 5 = {manhattan_norm:.4f}")

# 3. Норма максимуму (L∞-норма) - максимальне абсолютне значення
max_norm = np.linalg.norm(v, np.inf)
print(f"\n3. Норма максимуму (L∞): ||v||∞ = {max_norm:.4f}")
print(f"   Формула: max(|3|, |-4|, |5|) = max(3, 4, 5) = {max_norm:.4f}")

# Порівняння
print("\n" + "-" * 70)
print("ПОРІВНЯННЯ НОРМ:")
print(f"L2 (Евклідова):     {euclidean_norm:.4f}")
print(f"L1 (Мангеттенська): {manhattan_norm:.4f}")
print(f"L∞ (Максимум):      {max_norm:.4f}")

print("\n" + "-" * 70)
print("ВІДПОВІДЬ НА ПИТАННЯ:")
print("-" * 70)
print("\nКоли різниця між нормами буде СУТТЄВОЮ?")
print("\n1. Вектори з великим розкидом значень:")
print("   Наприклад: v = [1000, 1, 1]")
v_sparse = np.array([1000, 1, 1])
print(f"   L2: {np.linalg.norm(v_sparse, 2):.2f}")
print(f"   L1: {np.linalg.norm(v_sparse, 1):.2f}")
print(f"   L∞: {np.linalg.norm(v_sparse, np.inf):.2f}")
print("   → L1 і L∞ дуже близькі, але L2 значно менша")

print("\n2. Задачі машинного навчання:")
print("   - L1: створює розріджені рішення (багато нулів)")
print("   - L2: створює менші, але ненульові значення")
print("   - L∞: використовується для robust-методів")

print("\n3. Задачі оптимізації та регуляризації:")
print("   - L1 (Lasso): видаляє неважливі ознаки")
print("   - L2 (Ridge): зменшує всі ваги рівномірно")

print("\n\n" + "=" * 70)
print("ЗАДАЧА 2: НОРМАЛІЗАЦІЯ ВЕКТОРА")
print("=" * 70)

# Вектор w = [1, -2, 2, -1]
w = np.array([1, -2, 2, -1])
print(f"\nВектор w = {w}")

# Обчислюємо довжину (евклідову норму)
length_w = np.linalg.norm(w)
print(f"\nДовжина вектора w: ||w|| = {length_w:.4f}")
print(f"Формула: √(1² + (-2)² + 2² + (-1)²) = √(1 + 4 + 4 + 1) = √10 = {length_w:.4f}")

# Нормалізуємо вектор (ділимо на його норму)
w_normalized = w / length_w
print(f"\nНормалізований вектор: w_norm = {w_normalized}")
print(
    f"Покомпонентно: [{w_normalized[0]:.4f}, {w_normalized[1]:.4f}, {w_normalized[2]:.4f}, {w_normalized[3]:.4f}]"
)

# Перевіряємо, що норма = 1
norm_check = np.linalg.norm(w_normalized)
print("\nПеревірка норми нормалізованого вектора:")
print(f"||w_norm|| = {norm_check:.10f}")
print(f"Це дорівнює 1? {np.isclose(norm_check, 1.0)}")

print("\n" + "-" * 70)
print("ВІДПОВІДЬ НА ПИТАННЯ:")
print("-" * 70)
print("\nЧому нормалізація НЕ змінює напрямок вектора?")
print("\n1. Математично:")
print("   w_norm = w / ||w|| = (1/||w||) * w")
print("   Це множення на СКАЛЯР (число), яке тільки змінює довжину,")
print("   але не напрямок у просторі.")

print("\n2. Геометрично:")
print("   Уявіть стрілку в просторі:")
print("   - Оригінальний w: довга стрілка певного напрямку")
print("   - Нормалізований w_norm: та ж стрілка, але довжиною 1")
print("   → Обидві стрілки вказують в ОДИН БІК!")

print("\n3. Доказ через кут:")
cos_angle = np.dot(w, w_normalized) / (np.linalg.norm(w) * np.linalg.norm(w_normalized))
angle = np.arccos(cos_angle) * 180 / np.pi
print(f"   Кут між w і w_norm: {angle:.10f}° (має бути 0°)")

# Візуалізація
print("\n" + "=" * 70)
print("ВІЗУАЛІЗАЦІЯ")
print("=" * 70)

# Графік 1: Порівняння норм для різних векторів
fig = plt.figure(figsize=(15, 5))

# Створюємо кілька тестових векторів
test_vectors = [
    ([3, -4, 5], "v₁ = [3,-4,5]"),
    ([1, 1, 1], "v₂ = [1,1,1]"),
    ([10, 1, 1], "v₃ = [10,1,1]"),
    ([1, 10, 1], "v₄ = [1,10,1]"),
]

norms_data = {"L1": [], "L2": [], "L∞": []}
labels = []

for vec, label in test_vectors:
    v_test = np.array(vec)
    norms_data["L1"].append(np.linalg.norm(v_test, 1))
    norms_data["L2"].append(np.linalg.norm(v_test, 2))
    norms_data["L∞"].append(np.linalg.norm(v_test, np.inf))
    labels.append(label)

ax1 = fig.add_subplot(131)
x = np.arange(len(labels))
width = 0.25

ax1.bar(x - width, norms_data["L1"], width, label="L1 (Манхеттен)", color="coral")
ax1.bar(x, norms_data["L2"], width, label="L2 (Евклід)", color="steelblue")
ax1.bar(x + width, norms_data["L∞"], width, label="L∞ (Макс)", color="mediumseagreen")

ax1.set_xlabel("Вектори")
ax1.set_ylabel("Значення норми")
ax1.set_title("Порівняння різних норм", fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=15)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Графік 2: Візуалізація нормалізації (3D для w без останньої компоненти)
ax2 = fig.add_subplot(132, projection="3d")

# Використаємо перші 3 компоненти w для візуалізації
w_3d = w[:3]
w_norm_3d = w_normalized[:3]

# Оригінальний вектор
ax2.quiver(
    0,
    0,
    0,
    w_3d[0],
    w_3d[1],
    w_3d[2],
    color="red",
    arrow_length_ratio=0.15,
    linewidth=2.5,
    label="Оригінальний w",
)

# Нормалізований вектор
ax2.quiver(
    0,
    0,
    0,
    w_norm_3d[0],
    w_norm_3d[1],
    w_norm_3d[2],
    color="blue",
    arrow_length_ratio=0.3,
    linewidth=2.5,
    label="Нормалізований w",
)

ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")
ax2.set_title("Нормалізація НЕ змінює напрямок", fontweight="bold")
ax2.legend()
ax2.set_xlim([-3, 3])
ax2.set_ylim([-3, 3])
ax2.set_zlim([-3, 3])

# Графік 3: Одиничне коло (для 2D вектора)
ax3 = fig.add_subplot(133)

# Створюємо одиничне коло
theta = np.linspace(0, 2 * np.pi, 100)
circle_x = np.cos(theta)
circle_y = np.sin(theta)

ax3.plot(circle_x, circle_y, "k--", alpha=0.3, label="Одиничне коло")

# Візуалізуємо 2D вектор [3, -4]
v_2d = np.array([3, -4])
v_2d_norm = v_2d / np.linalg.norm(v_2d)

ax3.arrow(
    0,
    0,
    v_2d[0],
    v_2d[1],
    head_width=0.3,
    head_length=0.3,
    fc="red",
    ec="red",
    linewidth=2,
    label=f"Оригінал: {v_2d}",
)
ax3.arrow(
    0,
    0,
    v_2d_norm[0],
    v_2d_norm[1],
    head_width=0.1,
    head_length=0.1,
    fc="blue",
    ec="blue",
    linewidth=2,
    label="Нормалізований (на колі)",
)

ax3.set_xlim([-5, 5])
ax3.set_ylim([-5, 5])
ax3.set_aspect("equal")
ax3.grid(True, alpha=0.3)
ax3.set_xlabel("X")
ax3.set_ylabel("Y")
ax3.set_title("Нормалізація → проєкція на одиничне коло", fontweight="bold")
ax3.legend()
ax3.axhline(y=0, color="k", linewidth=0.5)
ax3.axvline(x=0, color="k", linewidth=0.5)

plt.tight_layout()
plt.show()

print("\nГотово! Перегляньте графіки для візуального розуміння.")
