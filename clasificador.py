import cv2 as cv
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle

# Carga todas las imágenes de una carpeta con su etiqueta
def load_images_from_folder(folder_path, label):
    images = []
    labels = []
    if not os.path.exists(folder_path):
        print(f"La carpeta {folder_path} no existe")
        return images, labels
    
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv.imread(img_path)
        if img is not None:
            images.append(img)
            labels.append(label)
    return images, labels

# Extrae histogramas de color en espacio HSV
def extract_color_histogram(img, bins=8):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hist_h = cv.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv.calcHist([hsv], [2], None, [bins], [0, 256])
    
    # Normalizar
    cv.normalize(hist_h, hist_h, 0, 1, cv.NORM_MINMAX)
    cv.normalize(hist_s, hist_s, 0, 1, cv.NORM_MINMAX)
    cv.normalize(hist_v, hist_v, 0, 1, cv.NORM_MINMAX)
    
    # Concatenar histogramas HSV
    features = np.concatenate([
        hist_h.flatten(), hist_s.flatten(), hist_v.flatten()
    ])
    return features

# Extrae características HOG
def extract_hog_features(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    all_features = []
    
    # HOG escala 1: Células pequeñas
    win_size = (32, 32)
    block_size = (8, 8)
    block_stride = (4, 4)
    cell_size = (4, 4)
    nbins = 9
    
    hog1 = cv.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    features1 = hog1.compute(img_gray).flatten()
    all_features.append(features1)
    
    # HOG escala 2: Células más grandes
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    
    hog2 = cv.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    features2 = hog2.compute(img_gray).flatten()
    all_features.append(features2)
    
    return np.concatenate(all_features).astype(np.float32)

# Extrae características estadísticas básicas
def extract_statistical_features(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Brillo y contraste
    brightness = np.mean(img_gray)
    contrast = np.std(img_gray)
    
    # Características por canal de color
    B, G, R = img[:,:,0], img[:,:,1], img[:,:,2]
    
    features = [
        brightness,
        contrast,
        np.mean(R), np.std(R),
        np.mean(G), np.std(G),
        np.mean(B), np.std(B)
    ]
    
    return np.array(features, dtype=np.float32)

# Extrae características combinadas
def extract_features(img):
    hog_features = extract_hog_features(img)
    
    # Histogramas de color HSV
    color_hist = extract_color_histogram(img, bins=48)
    
    # Características estadísticas
    stat_features = extract_statistical_features(img)
    
    # Combinar todas las características
    features = np.concatenate([
        hog_features,
        color_hist,
        stat_features
    ])
    
    return features

# Aplica data augmentation
def augment_image(img):
    augmented = [img]
    
    # Rotación +15°
    M = cv.getRotationMatrix2D((16, 16), 15, 1.0)
    rotated1 = cv.warpAffine(img, M, (32, 32))
    augmented.append(rotated1)
    
    # Rotación -15°
    M = cv.getRotationMatrix2D((16, 16), -15, 1.0)
    rotated2 = cv.warpAffine(img, M, (32, 32))
    augmented.append(rotated2)
    
    # Flip horizontal
    flipped = cv.flip(img, 1)
    augmented.append(flipped)
    
    # Brillo +20%
    bright = cv.convertScaleAbs(img, alpha=1.2, beta=0)
    augmented.append(bright)
    
    # Contraste +20%
    contrast = cv.convertScaleAbs(img, alpha=1.2, beta=-20)
    augmented.append(contrast)
    
    return augmented

# Carga los nombres de las clases desde archivo
def load_class_names(labels_file):
    class_names = []
    with open(labels_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

# Función principal
def main():
    USE_AUGMENTATION = True
    ML_MODEL = 'random_forest'  # Opciones: 'svm' o 'random_forest'
    
    print(f"Clasificador CIFAR-4 con {ML_MODEL.upper()}")
    print(f"Data Augmentation: {'ACTIVADO' if USE_AUGMENTATION else 'DESACTIVADO'}")
    
    # Rutas
    train_path = "./cifar10/train"
    test_path = "./cifar10/test"
    labels_file = "./cifar10/labels.txt"
    
    # Cargar nombres de las clases
    class_names = load_class_names(labels_file)
    print(f"Clases detectadas: {class_names}")
    
    # Cargar imágenes de entrenamiento
    print("Cargando imágenes de entrenamiento")
    train_images = []
    train_labels = []
    
    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(train_path, class_name)
        imgs, lbls = load_images_from_folder(class_path, idx)
        train_images.extend(imgs)
        train_labels.extend(lbls)
        print(f"  {class_name}: {len(imgs)} imágenes")
    
    
    # Aplicar data augmentation si está activado
    if USE_AUGMENTATION:
        print("Aplicando data augmentation")
        augmented_images = []
        augmented_labels = []
        
        for img, label in zip(train_images, train_labels):
            aug_imgs = augment_image(img)
            augmented_images.extend(aug_imgs)
            augmented_labels.extend([label] * len(aug_imgs))
        
        train_images = augmented_images
        train_labels = augmented_labels
        print(f"Total imágenes para el entrenamiento: {len(train_images)}")
    else:
        print(f"Total imágenes para el entrenamiento: {len(train_images)}")
    
    # Extraer features de entrenamiento
    print("Extrayendo características de entrenamiento")
    X_train = []
    for i, img in enumerate(train_images):
        features = extract_features(img)
        X_train.append(features)
        if (i + 1) % 500 == 0:
            print(f"  Procesadas {i + 1}/{len(train_images)} imágenes")
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(train_labels)
    
    print(f"Dimensión de características: {X_train.shape[1]}")
    
    # Normalizar características
    print("Normalizando características")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    # Entrenar modelo
    print(f"Entrenando clasificador")
    
    if ML_MODEL == 'svm':
        C_param = 1.0
        gamma_param = 'scale'
        classifier = SVC(C=C_param, kernel='rbf', gamma=gamma_param, verbose=True, cache_size=2000)
    elif ML_MODEL == 'random_forest':
        n_estimators = 200
        classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=30,
            min_samples_split=5,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
    else:
        raise ValueError(f"Modelo '{ML_MODEL}' no soportado. Usa 'svm' o 'random_forest'")
    
    classifier.fit(X_train, y_train)
    print("Entrenamiento completado")
    
    # Guardar el modelo y el scaler
    print("Guardando modelo")
    model_dir = f"models/{ML_MODEL}"
    os.makedirs(model_dir, exist_ok=True)
    
    with open(f'{model_dir}/model.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    with open(f'{model_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(f'{model_dir}/class_names.pkl', 'wb') as f:
        pickle.dump(class_names, f)
    print(f"Modelo guardado en: {model_dir}/")
    
    # Cargar imágenes de test
    print("Cargando imágenes de test")
    test_images = []
    test_labels = []
    
    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(test_path, class_name)
        imgs, lbls = load_images_from_folder(class_path, idx)
        test_images.extend(imgs)
        test_labels.extend(lbls)
        print(f"  {class_name}: {len(imgs)} imágenes")
    
    print(f"Total imágenes de test: {len(test_images)}")
    
    # Extraer características de test
    print("Extrayendo características de test")
    X_test = []
    for i, img in enumerate(test_images):
        features = extract_features(img)
        X_test.append(features)
        if (i + 1) % 200 == 0:
            print(f"  Procesadas {i + 1}/{len(test_images)} imágenes")
    
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(test_labels)
    
    # Normalizar test
    print("Normalizando características de test")
    X_test = scaler.transform(X_test)
    
    # Evaluar el modelo en test
    print("Evaluando en conjunto de test")
    
    test_pred = []
    batch_size = 500
    for i in range(0, len(X_test), batch_size):
        batch_pred = classifier.predict(X_test[i:i+batch_size])
        test_pred.extend(batch_pred)
        print(f"  Predichas {min(i+batch_size, len(X_test))}/{len(X_test)} imágenes")
    
    test_pred = np.array(test_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    print(f"Precisión en test: {test_accuracy * 100:.2f}%")
    
    # Mostrar el classification report
    print("Reporte de Clasificación")
    print(classification_report(y_test, test_pred, target_names=class_names))
    
    # Mostrar matriz de confusión simple
    print("Matriz de Confusión")
    confusion = np.zeros((len(class_names), len(class_names)), dtype=int)
    for true_label, pred_label in zip(y_test, test_pred):
        confusion[true_label][pred_label] += 1
    
    print(confusion)

if __name__ == "__main__":
    main()