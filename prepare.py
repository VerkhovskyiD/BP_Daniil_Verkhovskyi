# --- Import Knižnic ---
import os # Práca so súborovým systémom (priečinky, súbory)
import shutil # Kopírovanie súborov (na oddelenie anomálií)
import numpy as np # Práca s číslami a maticami
import pandas as pd # Práca s tabuľkami CSV
import matplotlib.pyplot as plt # Kreslenie
from PIL import Image # Otváranie a spracovanie obrázkov
from sklearn.preprocessing import StandardScaler # Normalizácia údajov
from sklearn.decomposition import PCA # Zníženie dimenzionality
from sklearn.manifold import TSNE # Rozšírená vizualizácia údajov
from sklearn.cluster import DBSCAN # Zhlukovací algoritmus bez učiteľa
from sklearn.metrics import silhouette_score # Hodnotenie kvality zhlukovania
from sklearn.neighbors import NearestNeighbors # Pre automatický výber parametra eps

# --- Nastavenia projektu ---
IMAGE_FOLDER = "images" # Priečinok, v ktorom sa nachádzajú obrázky, ktoré sa majú analyzovať
OUTPUT_FOLDER = "output/anomalies" # Kde sa nájdené anomálie budú ukladať
IMAGE_SIZE = (64, 64) # Veľkosť, na ktorú zmenšíme všetky obrázky

# --- Funkcia Nahrávanie a prevod obrázkov ---
def load_and_vectorize_image(folder_path, size=(64, 64)):
    vectors = [] # Zoznam, do ktorého budeme ukladať vektory čísel pre každý obrázok
    filenames = [] # Zoznam, do ktorého sa ukladajú názvy súborov
    for file in os.listdir(folder_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            try:
                path = os.path.join(folder_path, file)
                img = Image.open(path).convert("L").resize(size)
                vectors.append(np.array(img).flatten())
                filenames.append(file)
            except Exception as e:
                print(f"Chyba v súbore {file}: {e}")
    return np.array(vectors), filenames

# --- Funkcia zobrazenia anomálie ---
def show_anomalies_side_by_side(anomaly_files, folder_path, size=(64, 64)):
    count = len(anomaly_files)
    if count == 0:
        print("Neboli zistené žiadne anomálie.")
        return
    cols = 2
    rows = count
    plt.figure(figsize=(8, 3 * rows))
    for i, filename in enumerate(anomaly_files):
        img_path = os.path.join(folder_path, filename)
        original_img = Image.open(img_path)
        plt.subplot(rows, cols, i * 2 + 1)
        plt.imshow(original_img)
        plt.title(f"ORIGINÁL: {filename}", fontsize=8)
        plt.axis("off")
        processed_img = original_img.convert("L").resize(size)
        plt.subplot(rows, cols, i * 2 + 2)
        plt.imshow(processed_img, cmap='gray')
        plt.title("EDITOVANÉ", fontsize=8)
        plt.axis("off")
    plt.suptitle("Anomálne snímky: originál + spracovanie", fontsize=14)
    plt.tight_layout()
    plt.show()

# --- Tepelná mapa intenzity pixelov ---
def show_vector_heatmap(image_path, size=(64, 64)):
    try:
        img = Image.open(image_path).convert("L").resize(size)
        img_array = np.array(img)
        plt.figure(figsize=(6, 6))
        plt.imshow(img_array, cmap='hot', interpolation='nearest')
        plt.title("Tepelná mapa intenzity pixelov", fontsize=12)
        plt.colorbar(label="Hodnota odtieňa (jas)")
        plt.axis("off")
        plt.show()
    except Exception as e:
        print(f"⚠️ Nepodarilo sa zobraziť heatmapu: {e}")

# --- Funkcia ukladania anomálií ---
def save_anomalies(anomaly_filenames, src_folder, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    for file in anomaly_filenames:
        src_path = os.path.join(src_folder, file)
        dst_path = os.path.join(dest_folder, file)
        shutil.copyfile(src_path, dst_path)

# --- Funkcia automatického výberu eps ---
def find_best_eps(X_pca):
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(X_pca)
    distances, indices = neighbors_fit.kneighbors(X_pca)
    distances = np.sort(distances[:, 4])
    n_points = len(distances)
    all_coords = np.vstack((range(n_points), distances)).T
    first_point = all_coords[0]
    last_point = all_coords[-1]
    line_vec = last_point - first_point
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    vec_from_first = all_coords - first_point
    scalar_product = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    distances_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    best_idx = np.argmax(distances_to_line)
    eps_guess = distances[best_idx]
    print(f"Nájde sa optimálne 'koleno' v danej polohe {best_idx}  eps ≈ {eps_guess:.3f}")
    plt.figure(figsize=(10, 5))
    plt.plot(distances, label="5. najbližší sused")
    plt.scatter(best_idx, distances[best_idx], color='red', s=100, label="bod zlomu")
    plt.title("k-distance graf s vyznačeným bodom zlomu")
    plt.xlabel("Zoradené body")
    plt.ylabel("Vzdialenosť k 5. susedovi")
    plt.legend()
    plt.grid(True)
    plt.show()
    return eps_guess

# --- Hlavná funkcia analýzy ---
def run_analysis():
    X, filenames = load_and_vectorize_image(IMAGE_FOLDER, IMAGE_SIZE)
    print(f"Stiahnuto obrázkov: {len(X)}")
    if len(X) == 0:
        print("Žiadne obrázky na analýzu!")
        return
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Normalizované údaje")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print("PCA proces je dokončeny")
    eps_val = find_best_eps(X_pca)
    print(f"Automaticky vybrané eps = {eps_val:.3f}")
    min_samples_val = 3
    db = DBSCAN(eps=eps_val, min_samples=min_samples_val)
    labels = db.fit_predict(X_pca)
    print(f"Nájdené klastre.: {set(labels)}")
    if len(set(labels)) > 1 and -1 in labels:
        score = silhouette_score(X_pca, labels)
        print(f"Silhouette Score: {score:.3f}")
    else:
        print("Nie je dostatok klastrov na hodnotenie")
    os.makedirs("output", exist_ok=True)
    df = pd.DataFrame({"filename": filenames, "cluster": labels})
    df.to_csv("output/cluster_results.csv", index=False, encoding="utf-8-sig")
    print("CSV uložené: output/cluster_results.csv")
    anomalies = [filenames[i] for i, label in enumerate(labels) if label == -1]
    print(f" Zistené anomálie: {len(anomalies)}")
    print(anomalies)
    save_anomalies(anomalies, IMAGE_FOLDER, OUTPUT_FOLDER)
    show_anomalies_side_by_side(anomalies, IMAGE_FOLDER)
    if anomalies:
        print("🧊 Zobrazenie tepelnej mapy prvej anomálie:")
        show_vector_heatmap(os.path.join(IMAGE_FOLDER, anomalies[0]))
    tsne = TSNE(n_components = 2, random_state = 42, perplexity = 5)
    X_tsne = tsne.fit_transform(X_scaled)
    plt.figure(figsize=(10,6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c = labels, cmap = 'rainbow', edgecolors = 'black')
    plt.title("t-SNE vizualizácia zhlukov")
    plt.xlabel("TSNE 1")
    plt.ylabel("TSNE 2")
    plt.grid(True)
    plt.show()

# --- Spustenie kodu ---
if __name__ == "__main__":
    run_analysis()
