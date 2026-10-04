import serial
import requests
import numpy as np
import cv2
import time
import os
import base64
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
import ncnn
from security.model_integrity import verificar_modelos
from security.esp32_check import exigir_flash_encryption
from security.audit import registrar_evento
from security import serial_guard

# ── Configuracion ──
SERIAL_PORT  = "/dev/ttyACM0"
BAUD_RATE    = 921600
VPS_URL = "https://colmenaai.cl/api/deteccion"  # solo HTTPS
VPS_API_KEY = os.environ["RASPBERRY_API_KEY"]
HIVE_CODE = os.environ["RASPBERRY_HIVE_CODE"]
MODELO_ABEJA = "/home/polinocodificacion/modelo_ncnn_model"
MODELO_CORB  = "/home/polinocodificacion/modelo_corbicula_ncnn_model"
# El clasificador no es un modelo YOLO (es un ResNet18 exportado via pnnx), pero
# igual que los otros dos vive en una CARPETA (no archivos sueltos): asi
# _hash_directorio()/_permisos_inseguros() en security/model_integrity.py lo
# recorren correctamente (usan os.walk, que sobre un archivo suelto no anda).
MODELO_CLASIF = "/home/polinocodificacion/modelo_clasificador_ncnn"
MODELO_CLASIF_PARAM = os.path.join(MODELO_CLASIF, "model.ncnn.param")
MODELO_CLASIF_BIN   = os.path.join(MODELO_CLASIF, "model.ncnn.bin")
INTERVALO    = 5
REINTENTOS_CAMARA = 3
ABEJA_WORKERS = 2
CONF_ABEJA   = 0.55
CONF_CORB    = 0.55
PADDING      = 10

# Clases del clasificador, en el mismo orden que datasets.ImageFolder las
# indexo durante el entrenamiento (alfabetico): aromo=0, cerezo=1, yuyo=2.
CLASES_POLEN = ["aromo", "cerezo", "yuyo"]
CLASIF_SIZE = 224
# Deben coincidir exactamente con build_transforms() de train_classifier.py
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Seguridad ──
# ESP32_CHECK_MODO: "estricto" (produccion: aborta si Flash Encryption no esta verificado)
#                   "advertir" (solo pruebas de laboratorio)
ESP32_CHECK_MODO = os.environ.get("ESP32_CHECK_MODO", "estricto")
REVERIFICAR_MODELOS_CADA = 300  # segundos entre re-verificaciones de hash del modelo

COLOR_ABEJA_SIN_POLEN = (255, 0, 0)
COLOR_ABEJA_CON_POLEN = (0, 255, 0)
COLOR_CORBICULA       = (0, 255, 255)

GUARDAR_DIR = "/home/polinocodificacion/fotos_cascada"
PENDIENTES_DIR = "/home/polinocodificacion/fotos_pendientes"
os.makedirs(GUARDAR_DIR, exist_ok=True)
os.makedirs(PENDIENTES_DIR, exist_ok=True)

_modelos_corb_por_hilo = threading.local()
_vigilante_serial = serial_guard.VigilanteAnomalias()

def conectar_serial():
    for problema in serial_guard.verificar_puerto(SERIAL_PORT):
        registrar_evento("PUERTO_SERIAL_PERMISOS", problema)
    while True:
        try:
            # exclusive=True: ningun otro proceso puede abrir el puerto mientras el detector lo usa
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=15, exclusive=True)
            time.sleep(2)
            print(f"ESP32 conectado en {SERIAL_PORT}")
            return ser
        except Exception as e:
            print(f"Error conectando al ESP32: {e} - Reintentando en 5s...")
            time.sleep(5)

def obtener_foto(ser):
    for intento in range(1, REINTENTOS_CAMARA + 1):
        try:
            # Solo se envian comandos de la lista blanca
            ser.reset_input_buffer()  # descarta texto de arranque/depuracion del ESP32
            serial_guard.enviar_comando(ser, b"FOTO\n")

            # Respuesta de control acotada: SIZE:<n> o ERROR:<msg>
            linea_raw = serial_guard.leer_linea(ser)
            if linea_raw.startswith(b"ERROR:"):
                msg = linea_raw[6:].decode('ascii', errors='replace').strip()[:80]
                print(
                    f"ESP32 no pudo capturar la cámara "
                    f"(intento {intento}/{REINTENTOS_CAMARA}): {msg}"
                )
                continue
            try:
                size = serial_guard.validar_size(linea_raw)
            except serial_guard.RespuestaInvalida as e:
                _vigilante_serial.anomalia(f"{e}: {linea_raw[:40]!r}")
                continue
            print(f"  Recibiendo imagen de {size} bytes...")

            # Leer bytes de la imagen (nunca mas de `size`, ya acotado por MAX_JPEG_BYTES)
            data = b""
            while len(data) < size:
                chunk = ser.read(size - len(data))
                if not chunk:
                    print("Error: timeout leyendo imagen")
                    break
                data += chunk
            if len(data) != size:
                continue

            # Leer END
            serial_guard.leer_linea(ser)

            try:
                serial_guard.validar_jpeg(data)
            except serial_guard.RespuestaInvalida as e:
                _vigilante_serial.anomalia(str(e))
                continue

            # Convertir a imagen
            img_array = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                _vigilante_serial.anomalia("JPEG con firma válida pero no decodificable")
                continue
            _vigilante_serial.ok()
            return frame

        except Exception as e:
            print(f"Error obteniendo foto (intento {intento}/{REINTENTOS_CAMARA}): {e}")

    print("La cámara agotó los reintentos; se conservará la conexión serial")
    return None

def enviar_al_vps(resultado):
    try:
        payload = {
            **resultado,
            "hive_code": HIVE_CODE,
        }

        headers = {
            "X-API-Key": VPS_API_KEY,
            "Content-Type": "application/json",
        }

        response = requests.post(
            VPS_URL,
            json=payload,
            headers=headers,
            timeout=10,
        )

        if response.status_code == 200:
            print("  Datos enviados al VPS OK")
            return True
        else:
            print(
                f"  Error enviando al VPS: {response.status_code} "
                f"{response.text}"
            )
            return False

    except Exception as e:
        print(f"  Error conectando al VPS: {e}")
        return False


def _cargar_clasificador():
    """Carga (una vez por hilo) el clasificador NCNN del tipo polinico."""
    if not hasattr(_modelos_corb_por_hilo, "clasificador"):
        try:
            verificar_modelos([MODELO_CLASIF])
        except RuntimeError as e:
            registrar_evento("MODELO_ALTERADO_ANTES_DE_CARGAR", e)
            raise SystemExit(f"ABORTADO: {e}")
        net = ncnn.Net()
        net.load_param(MODELO_CLASIF_PARAM)
        net.load_model(MODELO_CLASIF_BIN)
        _modelos_corb_por_hilo.clasificador = net
    return _modelos_corb_por_hilo.clasificador


def clasificar_tipo_polen(recorte_corbicula_bgr):
    """Clasifica el tipo polinico de un recorte de corbicula (color BGR de OpenCV).

    Replica EXACTAMENTE el preprocesamiento usado en el entrenamiento
    (train_classifier.py / build_transforms): resize 224x224, RGB (no BGR!),
    escala 0-1, normalizado con las estadisticas de ImageNet. Si el orden de
    canales o la normalizacion no coinciden, el clasificador falla en
    silencio -- no tira error, simplemente clasifica mal, porque el color
    es la senal de la que depende toda la clasificacion.
    """
    net = _cargar_clasificador()

    rgb = cv2.cvtColor(recorte_corbicula_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (CLASIF_SIZE, CLASIF_SIZE), interpolation=cv2.INTER_LINEAR)
    img = resized.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1)).copy()  # HWC -> CHW, contiguo en memoria

    with net.create_extractor() as ex:
        ex.input("in0", ncnn.Mat(img).clone())
        _, out0 = ex.extract("out0")
        logits = np.array(out0)

    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    idx = int(probs.argmax())
    return CLASES_POLEN[idx], float(probs[idx])


def clasificar_abeja(tarea):
    """Detecta corbiculas en un recorte y clasifica el tipo polinico de cada
    una. Devuelve solo datos serializables (se ejecuta en un hilo)."""
    recorte, rx1, ry1, conf_a = tarea

    # Cada hilo mantiene su propia instancia para evitar inferencias concurrentes
    # sobre el mismo objeto de Ultralytics.
    if not hasattr(_modelos_corb_por_hilo, "modelo"):
        try:
            verificar_modelos([MODELO_CORB])  # justo antes de cargar (evita TOCTOU)
        except RuntimeError as e:
            registrar_evento("MODELO_ALTERADO_ANTES_DE_CARGAR", e)
            raise SystemExit(f"ABORTADO: {e}")  # el futuro propaga SystemExit al hilo principal
        _modelos_corb_por_hilo.modelo = YOLO(MODELO_CORB, task="detect")

    res_corb = _modelos_corb_por_hilo.modelo.predict(
        source=recorte, imgsz=640, conf=CONF_CORB, device="cpu", verbose=False
    )
    corbiculas = []
    for box_c in res_corb[0].boxes:
        cx1, cy1, cx2, cy2 = map(int, box_c.xyxy[0])
        recorte_corb = recorte[cy1:cy2, cx1:cx2]
        if recorte_corb.size == 0:
            continue

        tipo, conf_tipo = clasificar_tipo_polen(recorte_corb)

        corbiculas.append((
            rx1 + cx1,
            ry1 + cy1,
            rx1 + cx2,
            ry1 + cy2,
            float(box_c.conf[0]),
            tipo,
            conf_tipo,
        ))

    return conf_a, corbiculas


def procesar_frame(frame, model_abeja, foto_num, executor):
    h, w = frame.shape[:2]
    frame_anotado = frame.copy()

    res_abeja = model_abeja.predict(
        source=frame, imgsz=640, conf=CONF_ABEJA, device="cpu", verbose=False
    )

    abejas = res_abeja[0].boxes
    con_polen = sin_polen = total_corb = 0
    max_conf = 0.0
    tipos_polinicos = {tipo: 0 for tipo in CLASES_POLEN}

    tareas = []
    for box in abejas:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_a = float(box.conf[0])
        max_conf = max(max_conf, conf_a)

        rx1, ry1 = max(0, x1-PADDING), max(0, y1-PADDING)
        rx2, ry2 = min(w, x2+PADDING), min(h, y2+PADDING)
        recorte = frame[ry1:ry2, rx1:rx2]

        if recorte.size == 0:
            continue

        tareas.append((recorte, rx1, ry1, conf_a, x1, y1, x2, y2))

    futuros = [
        executor.submit(clasificar_abeja, tarea[:4])
        for tarea in tareas
    ]

    for tarea, futuro in zip(tareas, futuros):
        _, _, _, _, x1, y1, x2, y2 = tarea
        conf_a, corbiculas = futuro.result()
        tiene_polen = len(corbiculas) > 0

        if tiene_polen:
            color_abeja = COLOR_ABEJA_CON_POLEN
            label_abeja = f"bee+polen {conf_a:.2f}"
            con_polen += 1
        else:
            color_abeja = COLOR_ABEJA_SIN_POLEN
            label_abeja = f"bee {conf_a:.2f}"
            sin_polen += 1

        cv2.rectangle(frame_anotado, (x1,y1), (x2,y2), color_abeja, 2)
        cv2.putText(frame_anotado, label_abeja, (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_abeja, 1)

        for ax1, ay1, ax2, ay2, conf_c, tipo, conf_tipo in corbiculas:
            cv2.rectangle(frame_anotado, (ax1,ay1), (ax2,ay2), COLOR_CORBICULA, 2)
            cv2.putText(frame_anotado, f"{tipo} {conf_tipo:.2f}",
                        (ax1,ay1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_CORBICULA, 1)
            total_corb += 1
            tipos_polinicos[tipo] += 1

    # Guardar imagen solo si hay abeja con corbícula
    if con_polen > 0:
        nombre = f"{GUARDAR_DIR}/cascada_{foto_num:04d}_{time.strftime('%H%M%S')}.jpg"
        cv2.imwrite(nombre, frame_anotado)
        print(f"  Imagen guardada: {nombre}")

    # Convertir imagen a base64
    frame_small = cv2.resize(frame_anotado, (320, 240))
    _, buffer = cv2.imencode('.jpg', frame_small, [cv2.IMWRITE_JPEG_QUALITY, 50])
    imagen_b64 = base64.b64encode(buffer).decode('utf-8')

    total = con_polen + sin_polen
    distribucion = {
        "Con Polen": round(con_polen/total*100) if total > 0 else 0,
        "Sin Polen": round(sin_polen/total*100) if total > 0 else 0,
    }
    distribucion_polen = {
        tipo: round(cnt/total_corb*100) if total_corb > 0 else 0
        for tipo, cnt in tipos_polinicos.items()
    }

    return {
        "con_polen": con_polen,
        "sin_polen": sin_polen,
        "total_abejas": total,
        "corbiculas": total_corb,
        "confianza": max_conf if max_conf > 0 else 0.01,
        "distribucion": distribucion,
        "tipos_polinicos": tipos_polinicos,
        "distribucion_polen": distribucion_polen,
        "imagen": imagen_b64
    }

def capturar_imagenes(ser, cola, detener):
    foto_num = 0
    while not detener.is_set():
        frame = obtener_foto(ser)
        if frame is None:
            print(f"[{time.strftime('%H:%M:%S')}] Sin imagen, reintentando...")
            if not ser.is_open:
                ser = conectar_serial()
            else:
                time.sleep(1)
            continue

        ruta = os.path.join(
            PENDIENTES_DIR,
            f"captura_{int(time.time() * 1000)}_{foto_num:06d}.jpg",
        )
        if cv2.imwrite(ruta, frame):
            cola.put((ruta, foto_num))
            print(f"  Imagen en cola: {ruta} (pendientes: {cola.qsize()})")
            foto_num += 1
        else:
            print(f"  Error guardando imagen pendiente: {ruta}")

        time.sleep(INTERVALO)


def main():
    print("=" * 50)
    print("  Polinocodificacion - Detector USB Serial")
    print("=" * 50)

    # Integridad del modelo: hash SHA-256 + permisos, sin aceptar modelos externos
    try:
        verificar_modelos([MODELO_ABEJA, MODELO_CORB, MODELO_CLASIF])
    except RuntimeError as e:
        registrar_evento("MODELO_INTEGRIDAD_FALLIDA", e)
        raise SystemExit(f"ABORTADO: {e}")

    # Proteccion del firmware: Flash Encryption del ESP32-CAM (antes de abrir el puerto)
    exigir_flash_encryption(SERIAL_PORT, ESP32_CHECK_MODO)

    print("\nCargando modelo 1 (abejas)...")
    model_abeja = YOLO(MODELO_ABEJA, task="detect")
    print("Modelo de corbícula y clasificador de tipo polinico: se cargará una "
          "copia de cada uno por hilo al comenzar a clasificar")
    print("Modelos cargados OK")

    ser = conectar_serial()
    foto_num = 0

    # Esperar que el ESP32 esté listo
    linea = ser.readline().decode('utf-8', errors='ignore').strip()
    print(f"ESP32: {linea}")

    # La cola contiene rutas pequeñas; las imágenes viven en disco para no perderlas.
    cola = queue.Queue()
    detener = threading.Event()

    # Recuperar capturas que hayan quedado en disco por un reinicio o un fallo.
    for ruta in sorted(
        os.path.join(PENDIENTES_DIR, nombre)
        for nombre in os.listdir(PENDIENTES_DIR)
        if nombre.endswith(".jpg")
    ):
        cola.put((ruta, 0))
    if not cola.empty():
        print(f"Recuperadas {cola.qsize()} imágenes pendientes")

    hilo_captura = threading.Thread(
        target=capturar_imagenes,
        args=(ser, cola, detener),
        daemon=True,
    )
    hilo_captura.start()

    print(f"\nIniciando detección cada {INTERVALO} segundos...")
    print(f"Cola persistente en disco | Hilos por imagen: {ABEJA_WORKERS}")
    print("Ctrl+C para detener\n")

    ultima_verificacion = time.time()
    try:
        with ThreadPoolExecutor(max_workers=ABEJA_WORKERS) as executor:
            while True:
                # Re-verificacion periodica: el modelo no puede cambiar en caliente
                if time.time() - ultima_verificacion >= REVERIFICAR_MODELOS_CADA:
                    try:
                        verificar_modelos([MODELO_ABEJA, MODELO_CORB, MODELO_CLASIF])
                    except RuntimeError as e:
                        registrar_evento("MODELO_ALTERADO_EN_EJECUCION", e)
                        raise SystemExit(f"ABORTADO: {e}")
                    ultima_verificacion = time.time()
                try:
                    ruta, foto_num = cola.get(timeout=30)
                except queue.Empty:
                    continue
                inicio = time.time()
                try:
                    frame = cv2.imread(ruta)
                    if frame is None:
                        print(f"  No se pudo leer {ruta}; se conserva para revisión")
                        continue

                    resultado = procesar_frame(
                        frame, model_abeja, foto_num, executor
                    )
                    enviado = enviar_al_vps(resultado)
                    if enviado:
                        os.remove(ruta)
                    else:
                        print(f"  Se conserva la imagen pendiente: {ruta}")
                        cola.put((ruta, foto_num))

                    tiempo = time.time() - inicio
                    print(
                        f"[{time.strftime('%H:%M:%S')}] Abejas: {resultado['total_abejas']} | "
                        f"Con polen: {resultado['con_polen']} | Sin polen: {resultado['sin_polen']} | "
                        f"Corbículas: {resultado['corbiculas']} | Tiempo: {tiempo:.2f}s"
                    )
                except Exception as e:
                    print(f"Error procesando {ruta}: {e}; se conserva para reintento")
                    cola.put((ruta, foto_num))
                finally:
                    cola.task_done()
    except KeyboardInterrupt:
        print("\nDeteniendo detector...")
    finally:
        detener.set()
        try:
            ser.close()
        except Exception:
            pass

if __name__ == '__main__':
    main()
