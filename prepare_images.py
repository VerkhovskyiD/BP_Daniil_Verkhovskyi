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

# Zobrazenie všetkých položiek v priečinku
    for file in os.listdir(folder_path):
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"): #V samotnom priečinku hľadáme len fotografie
            # Celý proces spracovania každej fotografie
            try:
                path = os.path.join(folder_path, file) # Otvorenie obrázka
                img = Image.open(path).convert("L").resize(size) # Robime ho čiernobielym
                vectors.append(np.array(img).flatten()) # Komprimujteme na 64 × 64 a premeňame ho na dlhý vektor (4096 čísel)
                filenames.append(file) # Zapamätame si názov súboru.
            except Exception as e:
                print(f"Chyba v súbore {file}: {e}")

    return np.array(vectors), filenames # Vrátenie výsledku

# --- Funkcia zobrazenia anomálie ---
def show_anomalies_side_by_side(anomaly_files, folder_path, size=(64, 64)):

    # Kontrola počtu anomálií
    count = len(anomaly_files)
    if count == 0:
        print("Neboli zistené žiadne anomálie.")
        return
    # Označenie stĺpcov
    cols = 2
    rows = count
    plt.figure(figsize=(8, 3 * rows))

    # Prechádzame všetky anomálie. Otvortame ich a zobrazíme originál na ľavej strane a spracovanú verziu na pravej strane.
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

# --- Funkcia ukladania anomálií ---
def save_anomalies(anomaly_filenames, src_folder, dest_folder):
    # Ak priečinok neexistuje, vytvoríme ho.
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    # Kopirujeme každý súbor z images do output/anomalies
    for file in anomaly_filenames:
        src_path = os.path.join(src_folder, file)
        dst_path = os.path.join(dest_folder, file)
        shutil.copyfile(src_path, dst_path)

# --- Funkcia automatického výberu eps prostredníctvom grafu vzdialenosti ---
def find_best_eps(X_pca):

    # Pre každý bod nájdeme 5 najbližších susedov
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(X_pca)
    distances, indices = neighbors_fit.kneighbors(X_pca)
    distances = np.sort(distances[:, 4]) # Pozeráme sa len na vzdialenosti do piateho suseda

    # --- Algoritmus na automatické vyhľadávanie maximálneho ohybu grafu --

    n_points = len(distances) # Spočítajte celkový počet bodov na grafe vzdialenosti
    all_coords = np.vstack((range(n_points), distances)).T # Vytvorte dvojrozmerné pole: každý bod má súradnice (index, hodnota vzdialenosti).

    # Vezmime prvý a posledný bod grafu.
    first_point = all_coords[0]
    last_point = all_coords[-1]

    line_vec = last_point - first_point # Vypočítajte vektor spájajúci prvý a posledný bod.
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2)) # Normalizujte vektor (aby mal dĺžku 1).

    vec_from_first = all_coords - first_point # Pre každý bod vypočítame vektor od prvého bodu (first_point) k nemu.
    scalar_product = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1) # Nájdime skalárny súčin vektorov normalizovaným priamkovým vektorom.
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)  # Zostrojme vektorovú projekciu pre každý bod na priamke.
    vec_to_line = vec_from_first - vec_from_first_parallel # Vypočítajte vektor zo skutočného bodu do jeho priemetu na priamku.

    distances_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1)) # Nájdime euklidovskú vzdialenosť každého bodu od priamky.
    best_idx = np.argmax(distances_to_line) # Zistime index bodu, ktorý je najvzdialenejší od priamky.

    eps_guess = distances[best_idx] # Hodnotu vzdialenosti v nájdenom bode použijeme ako optimálny eps pre DBSCAN.
    print(f"Nájde sa optimálne 'koleno' v danej polohe {best_idx}  eps ≈ {eps_guess:.3f}")

    # Vykreslenie grafu a bodu zlomu
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
    X, filenames = load_and_vectorize_image(IMAGE_FOLDER, IMAGE_SIZE) # Nahrávanie obrázkov
    print(f"Stiahnuto obrázkov: {len(X)}")

    if len(X) == 0:
        print("Žiadne obrázky na analýzu!")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) # Normalizácia údajov
    print("Normalizované údaje")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled) # Zníženie dimenzionality pomocou PCA
    print("PCA proces je dokončeny")

    eps_val = find_best_eps(X_pca) # Vyhľadávanie automatických eps
    print(f"Automaticky vybrané eps = {eps_val:.3f}")
    min_samples_val = 3

    db = DBSCAN(eps=eps_val, min_samples=min_samples_val) # Spustenie DBSCAN
    labels = db.fit_predict(X_pca)
    print(f"Nájdené klastre.: {set(labels)}")

    if len(set(labels)) > 1 and -1 in labels:
        score = silhouette_score(X_pca, labels)
        print(f"Silhouette Score: {score:.3f}")
    else:
        print("Nie je dostatok klastrov na hodnotenie")

    # Uloženie tabuľky CSV
    os.makedirs("output", exist_ok=True)
    df = pd.DataFrame({"filename": filenames, "cluster": labels})
    df.to_csv("output/cluster_results.csv", index=False, encoding="utf-8-sig")
    print("CSV uložené: output/cluster_results.csv")

    # Výstup nájdených zhlukov a anomálií
    anomalies = [filenames[i] for i, label in enumerate(labels) if label == -1]
    print(f" Zistené anomálie: {len(anomalies)}")
    print(anomalies)

    save_anomalies(anomalies, IMAGE_FOLDER, OUTPUT_FOLDER)
    show_anomalies_side_by_side(anomalies, IMAGE_FOLDER)

    # Vizualizácia zhlukov pomocou t-SNE
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