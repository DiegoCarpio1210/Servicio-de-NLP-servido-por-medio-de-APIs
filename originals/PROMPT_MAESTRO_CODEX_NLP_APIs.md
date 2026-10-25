# ENCARGO PARA CODEX: SERVICIO DE NLP CON API — INGENIERÍA DE LA IA

Actúa como ingeniero sénior de machine learning, NLP, MLOps, Python, testing y despliegue local. Debes IMPLEMENTAR, VERIFICAR Y DOCUMENTAR un proyecto universitario funcional de **extractive question answering con preguntas sin respuesta (SQuAD 2.0)**. Tu trabajo es crear un repositorio reproducible, no solo dar sugerencias o escribir pseudocódigo. El servicio deberá funcionar en CPU en una laptop Windows, exponerse mediante túnel HTTPS y superar una evaluación automática. Puedes usar Codex como herramienta de desarrollo, pero **no se permiten APIs comerciales para inferencia ni modelos decoder GPT en el servicio**.

## 1. FUENTES OBLIGATORIAS Y ORDEN DE PRIORIDAD

En la carpeta de trabajo encontrarás los cuatro archivos originales de la tarea (o el ZIP del que debes extraerlos):

1. `Guía_de_trabajo_APIs_and_NLP.pdf`: 12 páginas; lee todas las secciones I–XV.
2. `qa_api_contract.yaml`: especificación OpenAPI **autoritaria cuando haya discrepancias con el PDF**; no la modifiques para hacer pasar pruebas.
3. `public_sample.json`: exactamente cinco casos públicos para probar el servicio, dos sin respuesta y uno con contexto largo; **no usar para entrenar ni memorizar respuestas**.
4. `REPORT_TEMPLATE.md`: plantilla obligatoria de informe, con encabezados y campos específicos.

Empieza verificando que pudiste leer los cuatro archivos y redacta un inventario de requisitos con criterios de aceptación. Si algunos datos personales/operativos faltan (ID del equipo, integrantes, hardware, URL de repositorio, token, URL de túnel), usa parámetros de configuración y marcadores claramente identificados; NO los inventes. Trabaja sin detenerte innecesariamente a solicitar esos valores.

## 2. RESTRICCIONES NO NEGOCIABLES

- Modelo: **encoder preentrenado BASE**, entrenado por nosotros mediante fine-tuning real para SQuAD 2.0. Prohibidos checkpoints ya afinados para SQuAD y modelos decoder como motor de servicio. Recomiendo evaluar `distilbert/distilbert-base-uncased` (o el ID base oficial vigente y comprobado) frente a BERT-base, privilegiando inferencia rápida en CPU. Explica las ventajas/desventajas y elige uno. No cambies silenciosamente a un checkpoint ya entrenado para preguntas y respuestas.
- Dataset: descarga `rajpurkar/squad_v2` de Hugging Face (splits train y validation). Debe existir la opción de usar una submuestra representativa y reproducible si el hardware es limitado; registra tamaño exacto y proporción de preguntas sin respuesta. No mezcles validación con entrenamiento. Incluye semilla fija.
- Nada de respuestas fijas para los cinco ejemplos, consultas externas de IA, endpoints de cloud hosting, o resultados ficticios.
- El entrenamiento puede usar GPU local o de la universidad; **el serving debe ejecutarse en la laptop, en CPU**, puerto **8000**, un único proceso/worker de modelo salvo medición que justifique otra cosa.
- Nunca escribas ni confirmes en logs ni commits el token. Se obtiene exclusivamente de la variable de entorno `QA_API_TOKEN`; si falta o está vacío, no permitir acceso a rutas protegidas y mostrar un error de configuración seguro. Debe ser fácil cambiarlo/reiniciar para el token nuevo del día de presentación.
- Conserva historial real de Git con varios commits lógicos; no hagas squash, no subas pesos a Git. `submission` será un tag en el commit final realmente validado; no simules un tag definitivo si todavía faltan entrenamiento, mediciones o datos obligatorios.
- Todo el repositorio, README, reporte, scripts y materiales de presentación deben estar en **inglés**, siguiendo la guía. Al comunicarte conmigo, explica el progreso en español claro.

## 3. ENTRENAMIENTO REAL Y EVALUACIÓN

Implementa scripts completos, configuración editable, dependencias fijadas razonablemente, instrucciones de entorno virtual para Windows PowerShell y uso opcional de GPU. Aplica tokenización rápida con `offset_mapping`, `return_overflowing_tokens`, `overflow_to_sample_mapping`, `truncation="only_second"`, `max_seq_length` y `doc_stride`. Asegura correspondencia correcta entre caracteres del contexto original y posiciones start/end de tokens. Maneja preguntas sin respuesta mediante posición CLS (o equivalente documentado) y trata correctamente ventanas donde la respuesta queda fuera de esa ventana.

Ajusta realmente el encoder y su cabeza QA; registra hiperparámetros, epochs, batch, learning rate, scheduler, warmup, semilla, hardware, tiempo, logs de entrenamiento/validación y gráfica de pérdida real (`logs/loss.png` o equivalente). Proporciona carga/reanudación de checkpoints y guarda pesos/tokenizador **fuera del Git versionado**, con una ruta configurable para servirlos. Diseña scripts de evaluación de SQuAD 2.0 con métricas EM/F1, métricas para answerable/unanswerable y prueba reproducible de selección del umbral de abstención **usando solo validación, no los cinco ejemplos públicos**.

Para decidir `null`, compara la mejor puntuación de span válido contra la puntuación de no-respuesta usando la lógica adecuada del modelo y las ventanas; calibra y documenta un umbral de decisión con resultados de validación. Evita sesgos por múltiples ventanas. Elige spans válidos con límites start <= end, longitud razonable, tokens del contexto (no de pregunta ni especiales), y descarta spans mal formados. Selecciona la mejor respuesta entre ventanas y respeta exactamente el fragmento del contexto sin reconstruirlo con tokenizer.decode.

## 4. INFERENCIA ROBUSTA PARA CONTEXTOS LARGOS

El contexto del API puede tener de 1 a 5000 caracteres y superar el límite de tokens del encoder. Implementa **sliding windows con solapamiento**, agrega resultados de todas las ventanas e incluye el final del texto. `sample-05` debe procesarse de verdad: no se permite cortar silenciosamente a 384 o 512 tokens y perder la respuesta final. Devuelve `start` como desplazamiento de **caracteres, base cero**, en el texto original. Comprueba SIEMPRE `context[start:start+len(answer)] == answer`; si no hay respuesta justificada, devuelve `answer: null` y `start: null` juntos. Contempla caracteres Unicode, respuestas repetidas y saltos de línea. `score` será una cifra real y finita entre 0 y 1 derivada razonablemente de los puntajes del modelo; su calibración perfecta no es exigida, pero no devuelvas constantes sin justificar.

El modelo y tokenizer deben cargarse una vez al iniciar el servicio, ponerse en modo evaluación, usar `torch.inference_mode()` y CPU durante serving, sin downloads en cada request. Considera un calentamiento inicial seguro y evita bloquear el servicio por precarga; si el modelo no está listo, `/health` no debe decir `ok`. Perfila y optimiza primero con mediciones, no con afirmaciones infundadas. El harness enviará **50 peticiones secuenciales**, **5 segundos máximo por petición**, **150 segundos globales** incluyendo túnel: apunta a latencias reales inferiores a estos límites, especialmente para textos largos, y advierte si el hardware impide cumplirlos.

## 5. API: CUMPLIMIENTO EXACTO DEL YAML

Implementa con FastAPI y uvicorn (u otra opción justificada), JSON UTF-8, puerto 8000:

- `GET /health`: **público**, `200 {"status":"ok"}` SOLO después de cargar completamente el modelo; mientras carga `503 {"status":"loading"}`.
- `GET /info`: **protegido**, `200` con `team` (ID registrado sin espacios, configurable), `model` (ID **del checkpoint BASE**), `params` (cuenta real de parámetros enteros > 0), opcionales `max_seq_length` y `doc_stride` reales.
- `POST /predict`: **protegido**, cuerpo estricto `{ "id": string (1–64), "context": string (1–5000), "question": string (1–500) }`, sin campos extra; `200 {"id": el_mismo_id, "answer": string_o_null, "start": integer_o_null, "score": numero_0_a_1}`.
- En `/info` y `/predict`, comprobar `Authorization: Bearer <token>` con comparación **exacta y sensible a mayúsculas/minúsculas** tanto para `Bearer` como para el token. Sin token, formato inválido o token incorrecto: **401** con `{ "error": "unauthorized" }`; incluir `WWW-Authenticate: Bearer` cuando corresponda. No usar comportamiento permisivo o automático del framework que cambie esta regla. `/health` nunca requiere token.
- JSON inválido, campos ausentes, longitudes inválidas, tipos inválidos o campos adicionales en `/predict`: **400 o 422** con cuerpo `{ "error": "bad_request", "detail": "..." }`. Sobrescribe si hace falta el manejador genérico de Pydantic/FastAPI: el formato por defecto `{"detail": [...]}` NO cumple el YAML. Fallos internos: 500 con `{ "error": "internal_error" }`, sin revelar secretos ni tracebacks a clientes. No inventar códigos adicionales.
- Siempre `Content-Type: application/json` para respuestas. El modelo debe devolver todos los campos obligatorios, no puede responder 200 con valores ilegales ni perder `id`.

## 6. PRUEBAS AUTOMÁTICAS Y CRITERIOS DE ACEPTACIÓN

Crea pruebas unitarias, integrales y de contrato. En las unitarias se pueden usar mocks claramente separados del servicio de producción; la demo real DEBE usar el checkpoint afinado.

Verifica: `/health` antes y después de carga; `/info` sin token, erróneo y correcto; `/predict` sin token, erróneo y correcto; exactitud de `Bearer`; solicitud mal formada; JSON roto; campos faltantes, extra, tipos incorrectos, vacíos y máximos; `id` idéntico; respuesta y offset fieles, incluidos Unicode y spans repetidos; `null` pareado; `score` en rango; pregunta imposible; múltiples ventanas y respuesta al final del contexto; tamaño y metadata genuinos; no existencia de secretos o pesos en Git. Cruza la estructura de peticiones/respuestas con `qa_api_contract.yaml` (validación por esquema cuando sea práctica).

Desarrolla `scripts/evaluate_public.py` que ejecute los **cinco** casos de `public_sample.json` a través del endpoint real, compruebe offsets, detecte las dos preguntas imposibles, verifique alternativas de respuesta permitidas y produzca una tabla por caso. No aceptes respuestas memorizadas ni optimices el umbral usando esos cinco. Mide latencia CPU de requests cortos y largos, y también por el túnel HTTPS. Indica tests aprobados, fallidos y no ejecutables: no los marques como aprobados sin ejecución.

## 7. DESPLIEGUE, RED Y REPRODUCIBILIDAD

Incluye `requirements.txt` o `pyproject.toml` reproducible, `.gitignore`, `.env.example` **sin secretos**, README detallado para Windows PowerShell, comandos de creación de entorno, instalación, descarga de dataset, entrenamiento GPU/CPU, evaluación, configuración `TEAM_ID`, `MODEL_DIR`, `QA_API_TOKEN`, arranque `uvicorn ... --host 0.0.0.0 --port 8000` y cinco pruebas `curl.exe`. Prepara `request.json` de ejemplo válido y un script de smoke-test. Indica cuánto disco/RAM requiere aproximadamente solo cuando lo puedas estimar de forma razonable.

Configura instrucciones para **ngrok** y una alternativa comprobable **Cloudflare Quick Tunnel** o Tailscale Funnel. El servicio DEBE ejecutarse en la laptop; el túnel solo lo expone con HTTPS. Define los pasos de probar desde una red externa, mantener el equipo conectado, desactivar suspensión, enviar URL al profesor antes del ensayo y exposición, y volver a enviarla si cambia. No afirmes haber probado un túnel si no puedes abrirlo realmente en el entorno.

## 8. REPOSITORIO, REPORTE Y PREPARACIÓN DE LA DEFENSA

Estructura el proyecto con módulos separados para preparación de datos, entrenamiento, métricas/calibración, inferencia, API, pruebas, scripts, configuración y documentación. Prefiere soluciones simples y comprobables, sin interfaz web ni funciones decorativas: el profesor solo llamará a la API.

Crea `REPORT.md` en la raíz usando **exactamente los encabezados 0–9, en el orden original de `REPORT_TEMPLATE.md`**, en inglés y con extensión aproximada 1–2 páginas (hasta ~1000 palabras de texto, sin contar tablas y figuras). No dejes ninguna sección vacía. Incluye resultados REALES de entrenamiento/validación, gráfico de pérdida, selección del umbral, cinco casos públicos, tamaño y latencia, arquitectura y túnel, modelo alternativo evaluado, limitaciones, equipo y uso de Codex. **La sección 7 exige al menos dos problemas REALES con causa, solución y commits reales**: documenta errores efectivamente encontrados durante la implementación y cómo los corregiste; no fabriques problemas ni hashes. La sección 8 compara decoder GPT o encoder–decoder con limitaciones específicas observadas y sus costos, solo como investigación escrita; NO implementar decoder.

Además, escribe una explicación de arquitectura comprensible para el equipo, un pitch de **3 minutos en inglés** que cubra los 4 criterios de la rúbrica (modelo, entrenamiento, servicio/acceso, comparación con decoder) y un documento de preparación de Q&A con preguntas que puedan hacerse a cualquier integrante sobre líneas/archivos/funciones reales del repositorio. El código debe quedar lo bastante claro para que lo podamos defender individualmente sin asistencia de IA durante la ronda de preguntas.

No publiques ni subas automáticamente el repositorio o pesos sin una ubicación/autorización definida. Cuando estén comprobados los entregables, explica cómo crear/publicar **varios commits reales**, colocar `git tag submission` en el commit final, excluir pesos y secretos, y compartir la URL. Si hay acceso a un repositorio Git y autorización, crea commits graduales auténticos; no hagas squash.

## 9. METODOLOGÍA DE EJECUCIÓN Y ENTREGA FINAL

Trabaja por etapas: (A) lectura íntegra, matriz de requisitos y plan; (B) proyecto y tests de contrato; (C) datos, fine-tuning y evaluación; (D) calibración y serving; (E) pruebas públicas, latencia, red; (F) documentación, informe, pitch y verificación final. Prioriza **modelo afinado real + validez de API + velocidad + reporte**, por encima de cualquier mejora visual. Ejecuta todos los pasos que permita el entorno sin esperar instrucciones adicionales. Si no dispones de GPU, Internet, token auténtico, datos del equipo o túnel, desarrolla y prueba todo lo que sí puedas, proporciona instrucciones exactas para completar lo bloqueado y **no inventes métricas ni afirmes que el servicio es funcional con pesos inexistentes**. Nunca sustituyas el fine-tuning por un checkpoint SQuAD preafinando o por respuestas fijas.

Al finalizar entrega: (1) árbol del proyecto y descripción de archivos; (2) checklist de **cada requisito** del PDF, YAML y REPORT_TEMPLATE marcado APROBADO, FALLIDO o PENDIENTE; (3) pruebas realmente ejecutadas con evidencia; (4) métricas genuinas y latencias si hay entrenamiento disponible; (5) órdenes exactas de Windows PowerShell para correr, entrenar y exponer el servicio; (6) tareas manuales pendientes: token profesor, ID del equipo, integrantes, GPU, URL Git y túnel; (7) riesgos de la rúbrica y un plan concreto para eliminarlos. El resultado debe estar listo para que cualquier miembro del equipo lo entienda y defienda, no solo para que «funcione en tu computadora».
