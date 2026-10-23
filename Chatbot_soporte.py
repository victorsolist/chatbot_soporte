# ============================================================
# Chatbot basado en reglas con NLTK
# Tema: Implementación de chatbots basados en reglas
# Entorno sugerido: Google Colab
# ============================================================

# Instalación de NLTK en caso de ser necesario
#!pip install nltk -q

import re
import random
import unicodedata
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer

# ------------------------------------------------------------
# 1. Configuración inicial de herramientas NLP
# ------------------------------------------------------------

tokenizer = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer("spanish")

def normalizar_texto(texto):
    """
    Convierte el texto a minúsculas, elimina acentos y caracteres especiales.
    """
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = texto.encode("ascii", "ignore").decode("utf-8")
    texto = re.sub(r"[^a-zñ0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

def tokenizar_y_stem(texto):
    """
    Tokeniza el texto y aplica stemming para reducir variaciones morfológicas.
    """
    texto = normalizar_texto(texto)
    tokens = tokenizer.tokenize(texto)
    stems = [stemmer.stem(token) for token in tokens]
    return tokens, stems

# ------------------------------------------------------------
# 2. Definición de intenciones del chatbot
# ------------------------------------------------------------

intenciones = {
    "saludo": {
        "ejemplos": ["hola", "buenos dias", "buenas tardes", "saludos"],
        "respuestas": [
            "Hola. Soy tu asistente de soporte técnico. Escribe ayuda para ver los temas.",
            "Bienvenido al asistente de soporte. Escribe ayuda para consultar el menú."]},
    "ayuda": {
        "ejemplos": ["ayuda", "menu", "opciones", "temas disponibles"],
        "respuestas": [
            "Puedo orientar sobre wifi, internet, contraseña, correo, impresora, ticket y seguimiento.",
            "Consulta wifi, internet, contraseña, correo, impresora, ticket o seguimiento. Escribe salir para terminar."]},
    "soporte_wifi": {
        "ejemplos": ["no conecta wifi", "red inalambrica", "wifi desconectado"],
        "respuestas": [
            "Revisa que el wifi esté activo y elige la red autorizada. Si persiste, reporta el mensaje de error a soporte.",
            "Comprueba el wifi y la red seleccionada. Si sigue sin conectar, solicita ayuda a soporte con el error observado."]},
    "soporte_internet": {
        "ejemplos": ["no tengo internet", "no puedo navegar", "navegacion lenta"],
        "respuestas": [
            "Comprueba si fallan varias páginas y si tu equipo está conectado a la red. Si persiste, reporta el alcance a soporte.",
            "Verifica la conexión de tu equipo y prueba otra página. Informa a soporte si la falla afecta a más personas."]},
    "recuperar_contrasena": {
        "ejemplos": ["olvide mi contrasena", "recuperar clave", "restablecer password", "acceso cuenta bloqueada"],
        "respuestas": [
            "Usa la recuperación oficial de tu organización. Si no puedes validar tu identidad, contacta a soporte. No compartas contraseñas ni códigos aquí.",
            "Solicita a soporte el procedimiento oficial de recuperación. Este asistente no restablece cuentas ni necesita tu contraseña."]},
    "soporte_correo": {
        "ejemplos": ["no abre outlook", "error correo electronico", "email no envia"],
        "respuestas": [
            "Comprueba tu conexión e intenta entrar al correo por la web oficial. Si falla, reporta el mensaje de error a soporte.",
            "Prueba el acceso al correo desde su web oficial. Si continúa el error, informa a soporte qué aplicación utilizas y qué mensaje aparece."]},
    "soporte_impresora": {
        "ejemplos": ["impresora no imprime", "error impresion", "necesito imprimir"],
        "respuestas": [
            "Revisa encendido, papel y la impresora seleccionada. Si hay un error, reporta el nombre del equipo y el mensaje.",
            "Comprueba que la impresora esté encendida, tenga papel y sea la seleccionada. Si persiste, solicita apoyo a soporte."]},
    "crear_ticket": {
        "ejemplos": ["crear ticket", "levantar reporte", "registrar incidencia", "necesito soporte"],
        "respuestas": [
            "Para solicitar atención sobre {servicio}, envía a soporte el equipo, problema, hora e impacto. Este prototipo no crea tickets.",
            "Reporta a soporte el equipo, la hora y el impacto del problema en {servicio}. El registro real debe hacerse por el canal oficial."]},
    "consultar_seguimiento": {
        "ejemplos": ["seguimiento", "estado de mi folio", "consultar avance"],
        "respuestas": [
            "Contacta a la mesa de soporte con tu folio. Este prototipo no consulta el estado real de los tickets.",
            "Para conocer el avance de tu atención, consulta el canal oficial de soporte e indica tu folio."]},
    "despedida": {
        "ejemplos": ["gracias", "hasta luego", "adios", "salir", "nos vemos"],
        "respuestas": [
            "Gracias por consultar. La conversación ha terminado.",
            "Hasta pronto. Puedes iniciar otra conversación cuando necesites orientación."]}
}

# ------------------------------------------------------------
# 3. Preparación de vocabulario por intención
# ------------------------------------------------------------

def construir_vocabulario_intenciones(intenciones):
    """
    Construye una representación basada en stems para cada intención.
    """
    vocabulario = {}

    for nombre_intencion, datos in intenciones.items():
        stems_intencion = []

        for ejemplo in datos["ejemplos"]:
            _, stems = tokenizar_y_stem(ejemplo)
            stems_intencion.extend(stems)

        vocabulario[nombre_intencion] = set(stems_intencion)

    return vocabulario

vocabulario_intenciones = construir_vocabulario_intenciones(intenciones)

# ------------------------------------------------------------
# 4. Detección de intención
# ------------------------------------------------------------

def detectar_intencion(mensaje):
    """
    Detecta la intención comparando stems del mensaje con el vocabulario
    de cada intención. Devuelve intención y puntuación.
    """
    _, stems_mensaje = tokenizar_y_stem(mensaje)
    stems_mensaje = set(stems_mensaje)

    puntajes = {}

    for intencion, stems_intencion in vocabulario_intenciones.items():
        coincidencias = stems_mensaje.intersection(stems_intencion)
        puntajes[intencion] = len(coincidencias)

    mejor_intencion = max(puntajes, key=puntajes.get)
    mejor_puntaje = puntajes[mejor_intencion]

    if mejor_puntaje == 0:
        return "desconocida", 0

    return mejor_intencion, mejor_puntaje

# ------------------------------------------------------------
# 5. Detección básica de entidades
# ------------------------------------------------------------

catalogo_servicios = {
    "wifi": ["wifi", "red inalambrica"],
    "internet": ["internet", "navegar", "navegacion"],
    "cuenta": ["cuenta", "usuario", "contrasena", "clave", "password"],
    "correo": ["correo", "outlook", "email"],
    "impresora": ["impresora", "imprimir", "impresion"],
    "mesa de ayuda": ["ticket", "reporte", "incidencia", "folio"]
}

def detectar_entidades(mensaje):
    """
    Detecta entidades básicas mediante diccionario de términos.
    """
    texto = normalizar_texto(mensaje)
    entidades = {}

    for entidad, palabras_clave in catalogo_servicios.items():
        for palabra in palabras_clave:
            if palabra in texto:
                entidades["servicio"] = entidad
                return entidades

    entidades["servicio"] = "no especificado"
    return entidades

# ------------------------------------------------------------
# 6. Generación de respuesta
# ------------------------------------------------------------

def generar_respuesta(mensaje):
    """
    Genera una respuesta considerando intención y entidades.
    """
    intencion, puntaje = detectar_intencion(mensaje)
    entidades = detectar_entidades(mensaje)

    if intencion == "desconocida":
        return {
            "intencion": intencion,
            "puntaje": puntaje,
            "entidades": entidades,
            "respuesta": (
                "No identifico una intención única. Describe un solo "
                     "problema o escribe ayuda para ver los temas."
            )
        }

    plantilla = random.choice(intenciones[intencion]["respuestas"])
    respuesta = plantilla.format(**entidades)

    return {
        "intencion": intencion,
        "puntaje": puntaje,
        "entidades": entidades,
        "respuesta": respuesta
    }

# ------------------------------------------------------------
# 7. Prueba interactiva del chatbot
# ------------------------------------------------------------

def iniciar_chatbot():
    """Recibe consultas y devuelve el historial como tabla de pandas."""
    historial = []

    print("Chatbot: Soy tu asistente de soporte. Escribe ayuda o salir.")
    print("Chatbot: Indicame cual es tu falla.")
    print("Chatbot: Escribe 'salir' para terminar.\n")

    while True:
        mensaje_usuario = input("Usuario: ")

        if normalizar_texto(mensaje_usuario) == "salir":
            print("Chatbot: Gracias por comunicarte. ¡Hasta pronto!")
            break

        resultado = generar_respuesta(mensaje_usuario)

        historial.append({
            "mensaje_usuario": mensaje_usuario,
            "intencion_detectada": resultado["intencion"],
            "puntaje": resultado["puntaje"],
            "entidades": resultado["entidades"],
            "respuesta": resultado["respuesta"]
        })

        print(f"Chatbot: {resultado['respuesta']}\n")

    return pd.DataFrame(historial)

# Ejecutar chatbot
historial_chat = iniciar_chatbot()
historial_chat

# ============================================================
# Evaluación básica del chatbot
# ============================================================

casos_prueba = [
    {"mensaje": "hola buenos dias", "intencion_esperada": "saludo"},
    {"mensaje": "ayuda", "intencion_esperada": "ayuda"},
    {"mensaje": "no conecta wifi", "intencion_esperada": "soporte_wifi"},
    {"mensaje": "navegacion lenta", "intencion_esperada": "soporte_internet"},
    {"mensaje": "gracias hasta luego", "intencion_esperada": "despedida"},
    {"mensaje": "quiero que me cambien mi pc", "intencion_esperada": "desconocida"}
]

resultados = []

for caso in casos_prueba:
    resultado = generar_respuesta(caso["mensaje"])
    resultados.append({
        "mensaje": caso["mensaje"],
        "intencion_esperada": caso["intencion_esperada"],
        "intencion_detectada": resultado["intencion"],
        "correcto": caso["intencion_esperada"] == resultado["intencion"],
        "respuesta": resultado["respuesta"]
    })

df_resultados = pd.DataFrame(resultados)

precision = df_resultados["correcto"].mean()
tasa_fallback = (df_resultados["intencion_detectada"] == "desconocida").mean()

print("Resultados de evaluación")
print("------------------------")
print(f"Precisión de intención: {precision:.2%}")
print(f"Tasa de fallback: {tasa_fallback:.2%}")

df_resultados

# Evaluación: ver pruebas_chatbot.py y la última sección del Colab.
# No se inicia una conversación al importar funciones para las pruebas.
if __name__ == "__main__":
    historial_chat = iniciar_chatbot()
    print(historial_chat.to_string(index=False))
