# Prueba desde tu teléfono con datos móviles

La prueba automática HTTPS ya terminó y su túnel está cerrado. La URL anterior NO sirve para esta prueba.
Hay que coordinar una nueva sesión temporal: la laptop mantiene API+túnel abiertos mientras pruebas y los
cierra al terminar. No se usará el token oficial. No hemos ejecutado estas acciones en tu teléfono.

## Preparar la laptop cuando estés listo

En PowerShell dentro del proyecto, genera un token nuevo solo para esta sesión:

```powershell
$env:QA_API_TOKEN = & .\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(24))"
$env:TEAM_ID = 'Pimientero'
$env:MODEL_DIR = (Resolve-Path 'models/continued20k').Path
Write-Host $env:QA_API_TOKEN
.\.venv\Scripts\python.exe -m uvicorn qa.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Lee el token únicamente en tu pantalla privada para introducirlo en el teléfono. No compartas capturas de
esa ventana, no actives transcripciones, no lo pegues en el chat ni en páginas web que envían peticiones por
un proxy. No se escribe a disco ni se incluye literalmente en el historial de comandos.
Abre otra PowerShell en el proyecto y, cuando decidas iniciar la sesión pública, ejecuta:

```powershell
.\tools\cloudflared.exe tunnel --url http://127.0.0.1:8000 --protocol http2
```

Copia la NUEVA URL HTTPS. Mantén ambas ventanas abiertas. No conectes la laptop al hotspot del mismo
teléfono: queremos que la laptop conserve su conexión habitual y el teléfono use su propia conexión móvil.

## Comprobaciones comunes

1. Desactiva Wi-Fi en el teléfono y activa datos móviles. Comprueba el indicador 4G/5G/LTE. Desactiva una VPN
   si conecta el teléfono a la misma red que la laptop; registra si utilizas otra VPN.
2. Abre en Chrome/Safari `https://NUEVA-URL/health`: debe aparecer `{"status":"ok"}`.
3. Abre `https://NUEVA-URL/info` sin token: debe aparecer `{"error":"unauthorized"}`. Es el comportamiento
   correcto, no un fallo. El navegador no suele mostrar el código HTTP: para confirmarlo usa el cliente abajo.
4. Para solicitudes con token usa una app local nativa; no un cliente web/proxy de terceros.

## Android: HTTP Shortcuts

Instala HTTP Shortcuts desde los enlaces de su [sitio oficial](https://http-shortcuts.rmy.ch/).
Su [guía](https://http-shortcuts.rmy.ch/shortcuts) explica métodos, encabezados y cuerpo de solicitud.

1. Pulsa `+`, elige un atajo HTTP normal, pon nombre `QA info`, método GET y URL `https://NUEVA-URL/info`.
2. En encabezados añade `Authorization` con valor `Bearer wrong`. Ejecuta: espera HTTP 401 y unauthorized.
3. Sustituye ese valor por `Bearer ` seguido del token temporal leído en la laptop. Respeta mayúscula B y
   un solo espacio. Ejecuta: espera HTTP 200, `team: Pimientero`, `training_steps:1600`, `training_run_steps:2534`.
4. Crea otro atajo `QA predict`, método POST, URL `https://NUEVA-URL/predict`, mismo Authorization y
   `Content-Type: application/json`. Selecciona cuerpo de texto JSON e introduce el cuerpo indicado abajo.
5. Ejecuta y guarda únicamente el estado HTTP y el JSON de respuesta, nunca capturas del encabezado secreto.
   No exportes/sincronices los atajos que contienen el token; elimínalos al terminar.

## iPhone: Atajos

Usa la app Atajos de Apple, siguiendo [Obtener contenido de URL](https://support.apple.com/en-euro/guide/shortcuts/apd58d46713f/ios).

1. Crea un atajo con `Solicitar entrada` para introducir el token en cada ejecución, sin guardarlo como texto
   literal dentro del atajo. No compartas ni exportes un atajo que contenga credenciales.
2. Añade `Obtener contenido de URL` con `https://NUEVA-URL/info`, método GET. Despliega sus opciones y añade
   el encabezado `Authorization`: escribe `Bearer ` e inserta la variable de la entrada solicitada.
3. Añade `Mostrar resultado`. Ejecuta primero introduciendo `wrong`: espera unauthorized (Atajos puede
   mostrarlo como error HTTP). Ejecuta con el token temporal: espera los datos de Pimientero y el paso 1600.
4. Duplica el atajo, cambia URL a `/predict` y método a POST. Usa cuerpo JSON con tres campos de tipo texto:
   `id`, `context` y `question`, exactamente como aparecen abajo. El cuerpo JSON debe enviarse como objeto,
   no como una cadena que contenga todo el JSON. Añade Content-Type: application/json si no está establecido.
5. Guarda el JSON de respuesta. Si tu versión de Atajos no muestra el estado HTTP, informa solo del cuerpo
   que observaste; no afirmes haber medido un código o tiempo que no viste. Borra el atajo temporal al terminar.

## Cuerpo sencillo para POST

```json
{"id":"phone-01","context":"Paris is the capital of France.","question":"What is the capital of France?"}
```

Comprueba que devuelve `id: phone-01` y los campos answer/start/score. Si responde Paris, start debe ser 0;
si responde null, start también debe ser null. Esta solicitud comprueba conectividad y formato: no es una
nueva evaluación de precisión ni modifica el modelo. Para comprobar el ejemplo público real, repite POST
usando exactamente los tres campos de request.json del repositorio (sample-01).

## Registrar y cerrar

Anota fecha/hora, Android/iPhone, Wi-Fi desactivado/datos móviles, códigos realmente observados y respuestas
sin encabezados. Puedes enviarnos esos resultados SIN el token. La prueba sigue pendiente hasta ver evidencia.
Estas solicitudes manuales no prueban el límite global de 50 solicitudes desde el teléfono; el ensayo automatizado
de 50 ya realizado fue HTTPS desde la laptop. No confundir los dos alcances.

Pulsa Ctrl+C primero en Cloudflare y después en uvicorn. En la ventana de la API ejecuta:

```powershell
Remove-Item Env:QA_API_TOKEN -ErrorAction SilentlyContinue
```

Elimina los atajos/encabezados temporales del teléfono. Vuelve a abrir `/health` desde datos móviles:
ya no debe devolver el JSON de nuestra API; Cloudflare puede mostrar un error. Anota también ese resultado.
