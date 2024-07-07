# By Miguel Gonzalez EGE HAINA  3 julio 2024
# Descripción:
# Este script utiliza un modelo BETO (BERT) para comparar una descripción de puesto
# con un grupo de CV de candidatos y seleccionar los que mas se ajustan a esta
# descripción del puesto.
#
# Prerequisitos:
# python 3.8
# pip3 install black tensorflow transformers scikit-learn pymupdf numpy
#
# Prerogrativas:
# -La selección esá sujeta a la calidad de la información contenida en el CV
# -El modelo debe estar pre-entranado en Español
# -fitz no debe ser instalado en el mismo virtual environment que pymupdf
#

import os
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import fitz  # PyMuPDF
import json
import logging
from concurrent.futures import ThreadPoolExecutor

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cargar el modelo BERT pre-entrenado en español y el tokenizador
logging.info("[*] Cargando el modelo...")
model_name = "dccuchile/bert-base-spanish-wwm-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModel.from_pretrained(model_name)

def load_keywords(json_file):
    """Cargar palabras clave y sus pesos desde un archivo JSON."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            keywords = json.load(f)
            logging.info(f"[*] Palabras clave cargadas: {keywords}")
            return keywords
    except Exception as e:
        logging.error(f"Error al cargar el archivo de palabras clave {json_file}: {e}")
        return {}

def embed_text(texto, keywords, default_weight=1.0):
    """Generar embeddings para el texto dado usando BERT, ponderando palabras clave."""
    try:
        inputs = tokenizer(texto, return_tensors="tf", padding=True, truncation=True)
        outputs = model(inputs)
        embeddings = outputs.last_hidden_state

        # Ponderar palabras clave
        for i, token in enumerate(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].numpy())):
            weight = keywords.get(token, default_weight)
            embeddings = tf.tensor_scatter_nd_update(embeddings, [[0, i]], embeddings[:, i] * weight)
            logging.debug(f"[*] Token: {token}, Peso: {weight}")

        return tf.reduce_mean(embeddings, axis=1).numpy().squeeze()
    except Exception as e:
        logging.error(f"Error al generar embeddings: {e}")
        return None

def load_text_from_file(file_path):
    """Cargar contenido de texto desde un archivo .txt o .pdf."""
    try:
        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        elif file_path.endswith(".pdf"):
            text = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
            return text
    except Exception as e:
        logging.error(f"Error al cargar el archivo {file_path}: {e}")
        return None
    return ""

def contains_keyword(text, keywords):
    """Verificar si el texto contiene alguna de las palabras clave."""
    for keyword in keywords:
        if (keyword.lower() in text) or (keyword.upper() in text) :
            return True
    return False

def load_cvs(directorio, keywords):
    """Cargar CVs desde el directorio especificado si contienen palabras clave."""
    cvs = {}
    try:
        for filename in os.listdir(directorio):
            if (filename.endswith(".txt") or filename.endswith(".pdf")) and (filename != os.path.basename(job_description_path)):
                file_path = os.path.join(directorio, filename)
                text = load_text_from_file(file_path)
                if text and contains_keyword(text, keywords):
                    cvs[filename] = text
    except Exception as e:
        logging.error(f"Error al cargar los CVs desde el directorio {directorio}: {e}")
    return cvs

def save_feedback(feedback_file, feedback_data):
    """Guardar la retroalimentación de los usuarios en un archivo JSON."""
    try:
        with open(feedback_file, 'w') as f:
            json.dump(feedback_data, f, indent=4)
    except Exception as e:
        logging.error(f"Error al guardar la retroalimentación: {e}")

def load_feedback(feedback_file):
    """Cargar la retroalimentación de los usuarios desde un archivo JSON."""
    try:
        with open(feedback_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def get_user_feedback(candidate):
    """Obtener la retroalimentación del usuario sobre un candidato."""
    while True:
        feedback = input(f"[?] ¿El candidato {candidate[0]} es relevante? (y/n): ").strip().lower()
        if feedback in ['y', 'n']:
            return feedback == 'y'
        print("Por favor, ingrese 'y' para sí o 'n' para no.")

def adjust_weights(similarities, feedback_data):
    """Ajustar los pesos de las similitudes basados en la retroalimentación."""
    for filename, similarity in similarities.items():
        if filename in feedback_data:
            if feedback_data[filename]:
                similarities[filename] *= 1.1  # Incrementar peso para relevantes
            else:
                similarities[filename] *= 0.9  # Reducir peso para no relevantes
    return similarities

feedback_file = 'feedback.json'
feedback_data = load_feedback(feedback_file)

def main(job_description_path, cvs_directory, keywords_file, top_n=5):
    logging.info("EGE HAINA - Este programa usa BERT para comparar CVs contra la descripción de puestos y seleccionar los que más se asemejan.")
    logging.info(f"[*] Directorio donde están los CVs: {cvs_directory}")
    logging.info(f"[*] Archivo con la descripción de puestos: {job_description_path}")

    desea_feedback = ''
    while desea_feedback not in ['y', 'n']:
        desea_feedback = input(f"[?] ¿Desea dar feedback para fines de entrenamiento? (y/n): ").strip().lower()

    try:
        # Cargar la descripción del trabajo
        logging.info("[*] Cargando la descripción de puesto")
        job_description = load_text_from_file(job_description_path)
        if job_description is None:
            raise ValueError("[!] La descripción de puesto no se pudo cargar.")
    except Exception as e:
        logging.error(f"Error al cargar la descripción de puesto: {e}")
        return

    # Cargar las palabras clave y sus pesos
    keywords = load_keywords(keywords_file)
    if not keywords:
        logging.error("[!] No se pudieron cargar las palabras clave.")
        return

    try:
        # Generar embedding para la descripción del trabajo
        logging.info("[*] Generando los embeddings de la descripción de puesto")
        job_desc_embedding = embed_text(job_description, keywords)
        if job_desc_embedding is None:
            raise ValueError("[!] El embedding de la descripción de puesto no se pudo generar.")
    except Exception as e:
        logging.error(f"Error al generar embedding para la descripción de puesto: {e}")
        return

    try:
        # Cargar CVs y generar embeddings
        logging.info("[*] Cargando y generando los CVs con sus embeddings")
        cvs = load_cvs(cvs_directory, keywords)
        logging.info(f"[*] CVs cargados: {len(cvs)}")
        if not cvs:
            raise ValueError("[!] No se encontraron CVs válidos en el directorio.")
        
        with ThreadPoolExecutor() as executor:
            cv_embeddings = {filename: embedding for filename, embedding in zip(cvs.keys(), executor.map(embed_text, cvs.values(), [keywords]*len(cvs))) if embedding is not None}
       
        if not cv_embeddings:
            raise ValueError("[!] No se pudieron generar embeddings para los CVs.")
    except Exception as e:
        logging.error(f"Error al cargar CVs o generar embeddings: {e}")
        return

    try:
        # Calcular similitudes coseno
        logging.info("[*] Calculando similitud de coseno con la descripción de puestos")
        similarities = {filename: cosine_similarity([job_desc_embedding], [embedding])[0][0]
                        for filename, embedding in cv_embeddings.items()}
        if not similarities:
            raise ValueError("[!] No se pudieron calcular las similitudes.")

        # Ajustar pesos basados en retroalimentación
        similarities = adjust_weights(similarities, feedback_data)

        # Obtener el tamaño de la muestra de los candidatos más idóneos
        while True:
            numero_entero = input(f"[?] Ingresa el tamaño de la muestra de los candidatos más idóneos [1 - {len(cvs)}]: ")
            try:
                top_n = int(numero_entero)
                if 1 <= top_n <= len(cvs):
                    break
                else:
                    logging.warning(f"[!] El número está fuera del rango 1 - {len(cvs)}. Inténtalo de nuevo.")
            except ValueError:
                logging.warning("[!] El valor ingresado no es un número entero válido. Inténtalo de nuevo.")

        # Ordenar CVs por similitud y seleccionar los mejores N
        logging.info("[*] Ordenando los candidatos")
        top_candidates = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:top_n]

        # Imprimir los mejores candidatos
        logging.info("\n[*] Resultados:")
        for i, candidate in enumerate(top_candidates, 1):
            print(f"[{i}] Candidate: {candidate[0]}, Similarity: {candidate[1]}")
            if desea_feedback == 'y':
                feedback_data[candidate[0]] = get_user_feedback(candidate)

        # Guardar la retroalimentación
        if desea_feedback == 'y':
            save_feedback(feedback_file, feedback_data)

    except Exception as e:
        logging.error(f"Error al calcular similitudes o ordenar candidatos: {e}")
    logging.info("\n[!] Fin_____")

if __name__ == "__main__":
    # Rutas a la descripción del trabajo y al directorio de CVs
    job_description_path = r"/home/mrgonzalez/Desktop/PYTHON/NLP_TENSORFLOW/job_description.txt"
    cvs_directory = r"/home/mrgonzalez/Desktop/PYTHON/NLP_TENSORFLOW/data/"
    keywords_file = r"/home/mrgonzalez/Desktop/PYTHON/NLP_TENSORFLOW/keywords_logistica.json"

    # Encontrar e imprimir los X mejores candidatos
    main(job_description_path, cvs_directory, keywords_file)
