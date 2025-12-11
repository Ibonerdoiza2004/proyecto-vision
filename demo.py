import cv2 as cv
import numpy as np
import pickle
import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from clasificador import extract_features, load_images_from_folder

# Carga el modelo entrenado
def load_model(model_type='svm'):
    model_dir = f"models/{model_type}"
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"No se encontró el modelo en {model_dir}")
    with open(f'{model_dir}/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(f'{model_dir}/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(f'{model_dir}/class_names.pkl', 'rb') as f:
        class_names = pickle.load(f)
    return model, scaler, class_names

# Evalúa el modelo y crea una confusion matrix
def evaluate_test_set(model, model_type, scaler, class_names):
    test_path = "./cifar10/test"
    print("Evaluando el modelo")
    
    y_true = []
    y_pred = []
    
    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(test_path, class_name)
        images, labels = load_images_from_folder(class_path, idx)
        
        if not images:
            continue
            
        print(f"  Prediciendo '{class_name}' ({len(images)} imágenes)")

        # Extraer características
        features_list = [extract_features(img) for img in images]
        
        # Convertir a numpy, normalizar y predecir
        X_batch = np.array(features_list, dtype=np.float32)
        X_batch = scaler.transform(X_batch)
        preds = model.predict(X_batch)
        
        y_true.extend(labels)
        y_pred.extend(preds)

    # Calcular métricas
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Precisión del modelo: {acc * 100:.2f}%")

    # Grafico
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha='right')
    plt.yticks(ticks, class_names)
    
    # Agregar números en cada celda
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10)
    
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta verdadera')
    plt.title(f'Matriz de Confusión (Precisión: {acc*100:.1f}%)')
    plt.tight_layout()
    os.makedirs(f'models/{model_type}', exist_ok=True)
    out_file = os.path.join(f'models/{model_type}', 'confusion_matrix.png')
    plt.savefig(out_file)
    plt.show()
    plt.close('all')

# Visualiza las características extraídas de una imagen
def visualize_features(img, true_label, predicted_label, confidence):
    img_size = 500
    spacing = 40
    canvas_width = img_size * 2 + spacing * 3
    canvas_height = img_size * 2 + spacing * 4 + 200
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    # Imagen original y etiquetas
    img_display = cv.resize(img, (img_size, img_size), interpolation=cv.INTER_NEAREST)
    canvas[spacing:spacing+img_size, spacing:spacing+img_size] = img_display
    
    y_text = spacing + img_size + 40
    cv.putText(canvas, f"Real: {true_label}", (spacing, y_text), 
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 150, 0), 2)
    cv.putText(canvas, f"Pred: {predicted_label}", (spacing, y_text+40), 
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv.putText(canvas, f"Conf: {confidence:.1f}%", (spacing, y_text+80), 
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    
    # Visualizacion HOG con gradientes
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    sobelx = cv.Sobel(img_gray, cv.CV_32F, 1, 0, ksize=3)
    sobely = cv.Sobel(img_gray, cv.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    magnitude_color = cv.applyColorMap(magnitude, cv.COLORMAP_JET)
    magnitude_display = cv.resize(magnitude_color, (img_size, img_size), interpolation=cv.INTER_NEAREST)
    
    x_col2 = spacing * 2 + img_size
    canvas[spacing:spacing+img_size, x_col2:x_col2+img_size] = magnitude_display
    cv.putText(canvas, "HOG", (x_col2, spacing+img_size+50), 
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    # Histograma de color HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    hist_canvas = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    bins = 48
    hist_h = cv.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv.calcHist([hsv], [2], None, [bins], [0, 256])
    
    cv.normalize(hist_h, hist_h, 0, 1, cv.NORM_MINMAX)
    cv.normalize(hist_s, hist_s, 0, 1, cv.NORM_MINMAX)
    cv.normalize(hist_v, hist_v, 0, 1, cv.NORM_MINMAX)
    
    hist_h_vis = hist_h * (img_size - 10)
    hist_s_vis = hist_s * (img_size - 10)
    hist_v_vis = hist_v * (img_size - 10)
    
    bin_w = img_size // bins
    for i in range(bins):
        x = i * bin_w
        cv.line(hist_canvas, (x, img_size), (x, img_size - int(hist_h_vis[i][0])), (255, 0, 0), 4)
        cv.line(hist_canvas, (x, img_size), (x, img_size - int(hist_s_vis[i][0])), (0, 255, 0), 3)
        cv.line(hist_canvas, (x, img_size), (x, img_size - int(hist_v_vis[i][0])), (0, 0, 255), 3)
    
    y_row2 = spacing * 3 + img_size + 120
    canvas[y_row2:y_row2+img_size, x_col2:x_col2+img_size] = hist_canvas
    cv.putText(canvas, "Histograma", (x_col2, y_row2+img_size+50), 
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    # Estadisticas de brillo y color
    stats_canvas = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    brightness = np.mean(img_gray)
    contrast = np.std(img_gray)
    
    B, G, R = img[:,:,0], img[:,:,1], img[:,:,2]
    
    stats_text = [
        f"Brillo: {brightness:.1f}",
        f"Contraste: {contrast:.1f}",
        "",
        f"R: {np.mean(R):.1f}+-{np.std(R):.1f}",
        f"G: {np.mean(G):.1f}+-{np.std(G):.1f}",
        f"B: {np.mean(B):.1f}+-{np.std(B):.1f}",
    ]
    
    for i, text in enumerate(stats_text):
        cv.putText(stats_canvas, text, (20, 60 + i*60), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    canvas[y_row2:y_row2+img_size, spacing:spacing+img_size] = stats_canvas
    cv.putText(canvas, "Estadisticas", (spacing, y_row2+img_size+50), 
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    return canvas

def main():
    ML_MODEL = 'svm'
    
    # Cargar modelo
    model, scaler, class_names = load_model(model_type=ML_MODEL)
    evaluate_test_set(model, ML_MODEL, scaler, class_names)
    test_path = "./cifar10/test"
    
    print(f"Demo visual del clasificador ({ML_MODEL.upper()})")
    print("Presiona 'q' para salir")
    
    # Seleccion aleatoria de imagenes
    while True:
        class_name = random.choice(class_names)
        class_path = os.path.join(test_path, class_name)
        
        images = os.listdir(class_path)
        img_name = random.choice(images)
        img_path = os.path.join(class_path, img_name)
        
        img = cv.imread(img_path)
        if img is None:
            continue
        
        features = extract_features(img)
        features = features.reshape(1, -1)
        features = scaler.transform(features)
        
        prediction = model.predict(features)[0]
        predicted_label = class_names[prediction]
        
        if hasattr(model, 'decision_function'):
            decision_scores = model.decision_function(features)[0]
            exp_scores = np.exp(decision_scores - np.max(decision_scores))
            probs = exp_scores / exp_scores.sum()
            confidence = probs[prediction] * 100
        elif hasattr(model, 'predict_proba'):
            probs = model.predict_proba(features)[0]
            confidence = probs[prediction] * 100
        else:
            confidence = 0.0
        
        visualization = visualize_features(img, class_name, 
                                          predicted_label, confidence)
        
        cv.imshow('Demo Visual - Clasificador CIFAR-4', visualization)
        
        key = cv.waitKey(0)
        if key == ord('q') or key == 27 or cv.getWindowProperty('Demo Visual - Clasificador CIFAR-4', cv.WND_PROP_VISIBLE) < 1:
            break
    
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
