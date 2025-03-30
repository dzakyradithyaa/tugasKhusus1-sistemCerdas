from sklearn.neighbors import KNeighborsClassifier
import numpy as np

X = np.array([
    [3.8, 90],
    [2.5, 70],
    [3.2, 85],
    [2.8, 60],
    [3.5, 88]
])
y = np.array(['Y', 'T', 'Y', 'T', 'Y'])

def get_valid_input(prompt, min_val, max_val):
    while True:
        try:
            value = float(input(prompt))
            if min_val <= value <= max_val:
                return value
            print(f"Input harus antara {min_val} dan {max_val}!")
        except ValueError:
            print("Harap masukkan angka yang valid!")

print("\n=== Prediksi Kelulusan Tepat Waktu ===")
print("Masukkan data mahasiswa baru:")
ipk = get_valid_input("IPK (skala 0-4): ", 0, 4)
kehadiran = get_valid_input("Persentase Kehadiran (%): ", 0, 100)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
prediction = knn.predict([[ipk, kehadiran]])

print(f"\nHasil Prediksi: {'LULUS (Y)' if prediction[0] == 'Y' else 'TIDAK LULUS (T)'}")
print("Berdasarkan 3 tetangga terdekat:")
distances, indices = knn.kneighbors([[ipk, kehadiran]])
for i, idx in enumerate(indices[0]):
    print(f"{i+1}. IPK: {X[idx][0]}, Kehadiran: {X[idx][1]}% → {y[idx]}")