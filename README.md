
# Automatická detekcia anomálií v obrazoch pomocou DBSCAN, PCA a t-SNE

Tento projekt predstavuje implementáciu systému na detekciu vizuálnych anomálií v množine obrázkov s využitím algoritmu DBSCAN (Density-Based Spatial Clustering of Applications with Noise). Ide o prístup učenia bez učiteľa (unsupervised learning), ktorý nevyžaduje označené dáta. Projekt je realizovaný v jazyku Python.

# Obsah repozitára

    | Súbor/Priečinok           | Popis                                                                 |
    |---------------------------|-----------------------------------------------------------------------|
    | `prepare_images.py`       | Hlavný skript na spracovanie obrázkov, zhlukovanie a detekciu anomálií|
    | `images/`                 | Priečinok s testovacími obrazmi                                       |
    | `output/`                 | Výstupné výsledky, vrátane CSV a extrahovaných anomálií               |
    | `pca_clusters.csv`        | Tabuľka s výsledkami zhlukovania pre každé obrázkové dáta             |

#  Použité technológie a knižnice

    - Python 3.x
    - `PIL` (Pillow) – spracovanie obrázkov
    - `numpy`, `pandas` – numerické a dátové operácie
    - `matplotlib` – vizualizácia dát
    - `scikit-learn` – DBSCAN, PCA, TSNE, normalizácia, Silhouette Score

#  Funkcionalita

    - Automatické načítanie a vektorovanie obrázkov z priečinka
    - Zníženie dimenzionality pomocou PCA (pre výpočet) a t-SNE (pre vizualizáciu)
    - Automatizovaný výpočet optimálneho parametra `eps` cez k-distance graf
    - Zhlukovanie pomocou DBSCAN a detekcia odľahlých hodnôt (anomalie)
    - Zobrazenie anomálnych obrázkov (originál + upravený)
    - Uloženie výsledkov do `.csv` a extrakcia anomálií do samostatného priečinka

#  Priebeh použitia

    1. Nahraj testovacie obrázky do priečinka `images/`
    2. Spusti skript `prepare_images.py`
    3. Výstupy sa uložia do priečinka `output/` a `pca_clusters.csv`
    4. Anomálie sa zobrazia a uložia samostatne

#  Ukážka výstupu

    - CSV súbor s identifikovanými zhlukmi
    - Zobrazenie anomálií (vľavo originál, vpravo upravený obrázok)
    - Vizualizácia pomocou PCA a t-SNE

🧪 Autor: Daniil Verkhovskyi
