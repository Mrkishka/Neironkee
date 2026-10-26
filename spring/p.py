# =========================
# 1. Импорт библиотек
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

plt.style.use('ggplot')


# =========================
# 2. Загрузка данных
# =========================
ds = pd.read_csv('Marketing_Data.csv')
print(ds.head())


# =========================
# 3. Корреляционная матрица
# =========================
plt.figure(figsize=(10, 6))
sns.heatmap(ds.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()


# =========================
# 4. Гистограммы
# =========================
ds.hist(figsize=(10, 8), edgecolor='black')
plt.suptitle("Feature Distributions")
plt.savefig('feature_histograms.png', dpi=150, bbox_inches='tight')
plt.show()


# =========================
# 5. Boxplot (выбросы)
# =========================
plt.figure(figsize=(8, 6))
sns.boxplot(data=ds)
plt.title("Boxplot (Outliers Detection)")
plt.savefig('boxplot_outliers.png', dpi=150, bbox_inches='tight')
plt.show()


# =========================
# 6. Удаление выбросов (3σ)
# =========================
cols = ['youtube', 'facebook', 'newspaper', 'sales']
mask = (ds[cols] - ds[cols].mean()).abs() <= (3 * ds[cols].std())
ds_clean = ds[mask.all(axis=1)].reset_index(drop=True)

print("Размер исходных данных:", ds.shape)
print("После очистки:", ds_clean.shape)


# =========================
# 7. Подготовка данных
# =========================
X = ds[['youtube', 'facebook', 'newspaper']]
y = ds['sales']

X_clean = ds_clean[['youtube', 'facebook', 'newspaper']]
y_clean = ds_clean['sales']


# =========================
# 8. Train/Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_clean, y_clean, test_size=0.4, random_state=42
)


# =========================
# 9. Масштабирование
#    scaler обучается только на чистых тренировочных данных,
#    затем применяется ко всем выборкам.
# =========================
scaler = StandardScaler()
scaler.fit(Xc_train)   # обучаем на чистых тренировочных данных

# Масштабируем все наборы
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

Xc_train_scaled = scaler.transform(Xc_train)
Xc_test_scaled  = scaler.transform(Xc_test)


# =========================
# 10. Обучение модели (только на чистых данных)
# =========================
model = LinearRegression()
model.fit(Xc_train_scaled, yc_train)


# =========================
# 11. Предсказания на обоих тестовых наборах
# =========================
y_pred_dirty = model.predict(X_test_scaled)   # тест с выбросами
y_pred_clean = model.predict(Xc_test_scaled)  # тест без выбросов


# =========================
# 12. Функция для метрик
# =========================
def get_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


rmse_dirty, mae_dirty, r2_dirty = get_metrics(y_test, y_pred_dirty)
rmse_clean, mae_clean, r2_clean = get_metrics(yc_test, y_pred_clean)


# =========================
# 13. Сравнение метрик
# =========================
results = pd.DataFrame({
    "Test data": ["With outliers", "Without outliers"],
    "RMSE": [rmse_dirty, rmse_clean],
    "MAE": [mae_dirty, mae_clean],
    "R2": [r2_dirty, r2_clean]
})

print("\nModel performance on different test sets (trained on clean data):")
print(results)


# =========================
# 14. Коэффициенты модели
# =========================
coef_df = pd.DataFrame({
    "Feature": ['youtube', 'facebook', 'newspaper'],
    "Coefficient": model.coef_
})

print("\nModel coefficients (trained on clean data):")
print(coef_df)
print("Intercept:", model.intercept_)


# =========================
# 15. Графики Actual vs Predicted (одна модель, два тестовых набора)
# =========================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Тест с выбросами ---
axes[0].scatter(y_test, y_pred_dirty, alpha=0.7, color='tomato', edgecolors='white', s=60)
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             linestyle='--', color='black')
axes[0].set_xlabel("Actual Sales")
axes[0].set_ylabel("Predicted Sales")
axes[0].set_title("Test with outliers")
axes[0].text(0.05, 0.93, f'RMSE = {rmse_dirty:.2f}', transform=axes[0].transAxes, fontsize=11)
axes[0].text(0.05, 0.86, f'R² = {r2_dirty:.3f}', transform=axes[0].transAxes, fontsize=11)

# --- Тест без выбросов ---
axes[1].scatter(yc_test, y_pred_clean, alpha=0.7, color='steelblue', edgecolors='white', s=60)
axes[1].plot([yc_test.min(), yc_test.max()],
             [yc_test.min(), yc_test.max()],
             linestyle='--', color='black')
axes[1].set_xlabel("Actual Sales")
axes[1].set_ylabel("Predicted Sales")
axes[1].set_title("Test without outliers")
axes[1].text(0.05, 0.93, f'RMSE = {rmse_clean:.2f}', transform=axes[1].transAxes, fontsize=11)
axes[1].text(0.05, 0.86, f'R² = {r2_clean:.3f}', transform=axes[1].transAxes, fontsize=11)

plt.suptitle("One model (trained on clean data) tested on different sets", fontsize=14)
plt.tight_layout()
plt.savefig('actual_vs_predicted_one_model.png', dpi=150, bbox_inches='tight')
plt.show()


# =========================
# 16. Кривая обучения (Learning curve)
# =========================
train_sizes, train_scores, test_scores = learning_curve(
    LinearRegression(), X_clean, y_clean, cv=5
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label="Train score")
plt.plot(train_sizes, test_mean, label="Validation score")

plt.xlabel("Training size")
plt.ylabel("Score")
plt.title("Learning Curve (clean data)")
plt.legend()
plt.savefig('learning_curve.png', dpi=150, bbox_inches='tight')
plt.show()