import numpy as np
import matplotlib.pyplot as plt

# Матриця
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

A_T = A.T

# Створення графіків
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Оригінальна матриця
im1 = axes[0].imshow(A, cmap='Blues', alpha=0.6)
axes[0].set_title('Оригінальна матриця A', fontsize=14, weight='bold')

# Додаємо значення в клітинки
for i in range(3):
    for j in range(3):
        axes[0].text(j, i, str(A[i, j]), ha='center', va='center', 
                     fontsize=20, weight='bold')

# Виділяємо перший рядок [1,2,3]
axes[0].add_patch(plt.Rectangle((-0.5, -0.5), 3, 1, 
                                fill=False, edgecolor='red', linewidth=3))
axes[0].text(1, -1, 'Рядок [1,2,3]', ha='center', fontsize=12, 
             color='red', weight='bold')

# Виділяємо перший стовпець [1,4,7]
axes[0].add_patch(plt.Rectangle((-0.5, -0.5), 1, 3, 
                                fill=False, edgecolor='green', linewidth=3))
axes[0].text(-1.2, 1, 'Стовпець\n[1,4,7]', ha='center', fontsize=10, 
             color='green', weight='bold')

axes[0].set_xticks(range(3))
axes[0].set_yticks(range(3))
axes[0].set_xlabel('Стовпці →', fontsize=11)
axes[0].set_ylabel('Рядки ↓', fontsize=11)

# Транспонована матриця
im2 = axes[1].imshow(A_T, cmap='Oranges', alpha=0.6)
axes[1].set_title('Транспонована матриця Aᵀ', fontsize=14, weight='bold')

# Додаємо значення
for i in range(3):
    for j in range(3):
        axes[1].text(j, i, str(A_T[i, j]), ha='center', va='center', 
                     fontsize=20, weight='bold')

# [1,2,3] тепер СТОВПЕЦЬ!
axes[1].add_patch(plt.Rectangle((-0.5, -0.5), 1, 3, 
                                fill=False, edgecolor='red', linewidth=3))
axes[1].text(-1.2, 1, 'Стовпець\n[1,2,3]', ha='center', fontsize=10, 
             color='red', weight='bold')

# [1,4,7] тепер РЯДОК!
axes[1].add_patch(plt.Rectangle((-0.5, -0.5), 3, 1, 
                                fill=False, edgecolor='green', linewidth=3))
axes[1].text(1, -1, 'Рядок [1,4,7]', ha='center', fontsize=12, 
             color='green', weight='bold')

axes[1].set_xticks(range(3))
axes[1].set_yticks(range(3))
axes[1].set_xlabel('Стовпці →', fontsize=11)
axes[1].set_ylabel('Рядки ↓', fontsize=11)

plt.tight_layout()
plt.show()

# Також можна показати як вектор повертається
fig2, ax = plt.subplots(figsize=(10, 6))

# Рядковий вектор (горизонтальний)
ax.arrow(0, 2, 3, 0, head_width=0.15, head_length=0.2, 
         fc='red', ec='red', linewidth=3, label='Рядок [1,2,3]')
ax.text(1.5, 2.3, '[1, 2, 3] →', fontsize=14, ha='center', color='red', weight='bold')

# Стовпцевий вектор (вертикальний)
ax.arrow(5, 0, 0, 3, head_width=0.15, head_length=0.2, 
         fc='blue', ec='blue', linewidth=3, label='Стовпець [1,2,3]ᵀ')
ax.text(5.5, 1.5, '↑\n[1]\n[2]\n[3]', fontsize=12, ha='left', color='blue', weight='bold')

# Стрілка транспонування
ax.annotate('', xy=(4.5, 1.5), xytext=(3.5, 1.5),
            arrowprops=dict(arrowstyle='->', lw=3, color='purple'))
ax.text(4, 1.8, 'Транспонування', fontsize=13, ha='center', 
        color='purple', weight='bold')

ax.set_xlim(-0.5, 6.5)
ax.set_ylim(-0.5, 3.5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_title('Як вектор повертається на 90°', fontsize=16, weight='bold')
ax.legend(fontsize=12)

plt.tight_layout()
plt.show()