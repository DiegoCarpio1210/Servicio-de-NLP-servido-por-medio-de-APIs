# Ejecución y ensayo — equipo Pimientero

Guía operativa para el equipo. El informe y la presentación académica permanecen en inglés.
No requiere entrenar otra vez. Un ensayo temporal de Cloudflare ya pasó desde la laptop y fue cerrado. La prueba desde otra red sigue pendiente; consulta PHONE_TEST_ES.md.

## 1. Iniciar la API en tu computadora

Abre PowerShell en la carpeta del proyecto. Pega este bloque y escribe el token oficial cuando se solicite.
El valor se oculta y queda en la memoria del proceso; no se escribe en el código ni en un archivo `.env`.

```powershell
Set-Location 'C:\Users\Lenovo P52s\Desktop\Servicio de NLP servido por medio de APIs'
$env:TEAM_ID = 'Pimientero'
$env:MODEL_DIR = (Resolve-Path 'models/quality40k').Path
$env:TORCH_NUM_THREADS = '4'
$qaSecret = Read-Host 'Token QA del profesor' -AsSecureString
$env:QA_API_TOKEN = [System.Net.NetworkCredential]::new('', $qaSecret).Password
Remove-Variable qaSecret
.\.venv\Scripts\python.exe -m uvicorn qa.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Mantén esa ventana abierta. No uses `--reload` ni varios workers. En una segunda ventana:

```powershell
curl.exe -i http://127.0.0.1:8000/health
```

Espera HTTP 200 y `{"status":"ok"}`. Mientras carga puede devolver 503. Si no inicia, revisa que existan
los pesos y calibration.json en models/quality40k y que el token no esté vacío. GitHub no contiene pesos.
La API debe mostrar en `/info` `training_steps:1600`, `training_run_steps:2534`, umbral -0.98309326171875 y
el hash que figura en logs/quality40k/info.json. No compartas el token en el chat.

## 2. Abrir HTTPS — solo cuando autorices la exposición

El ensayo automatizado autorizado con token aleatorio ya se ejecutó y se cerró. Los comandos siguientes permiten una nueva sesión coordinada. Si quieres que Codex
haga una prueba temporal, autoriza expresamente abrir el túnel y especifica si usarás un token temporal o el
oficial configurado localmente. La prueba desde otra red requiere además un segundo dispositivo/conexión.

En otra PowerShell dentro del proyecto, con la API lista:

```powershell
.\tools\cloudflared.exe tunnel --url http://127.0.0.1:8000
```

Copia la URL `https://...trycloudflare.com` que aparezca y deja esa ventana abierta. Cambia al reiniciar.
No envíes al profesor una URL antigua. No añadas login web/OAuth que impida acceder a su evaluador.

## 3. Probar desde otra red

Usa una segunda computadora conectada, por ejemplo, a datos móviles, fuera de la red de la computadora
que sirve la API. Clona el repositorio público allí; no necesitas descargar pesos ni datasets para este ensayo.
En la raíz del clon:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install httpx jsonschema pyyaml
.\.venv\Scripts\python.exe -m scripts.rehearse --url https://TU-URL.trycloudflare.com --external-network --network-note "Segundo equipo con datos moviles" --output logs/external_network_verification.json
```

Sustituye la URL por la real. El programa pide el token de forma oculta si no existe QA_API_TOKEN en esa
ventana. No pases el token como argumento. Usa un nombre de salida nuevo para cada ensayo.

El ensayo verifica primero health, accesos sin token/incorrectos y `/info`, incluido el hash. Luego envía
50 solicitudes secuenciales: cinco ejemplos públicos repetidos diez veces, con plazo real de 5 segundos
por solicitud y 150 segundos para el lote. Verifica JSON, IDs, offsets y nulls. No es el examen oculto ni
una nueva medición independiente de precisión. La comprobación previa no forma parte del cronómetro del lote.

El archivo separa `technical_pass` de `accuracy_pass`. Actualmente se espera que sample-04 siga siendo
incorrecto: el programa termina con código 1 si falla precisión o técnica; revisa ambas claves. No ignores
el código ni lo confundas automáticamente con un fallo de conexión. `--external-network` registra tu
declaración: el programa no puede certificar por sí mismo que cambiaste de red. Conserva el registro y anota
qué dispositivos/redes usaste. No se declarará realizado este ensayo hasta recibir evidencia real.

## 4. Instalar y probar ngrok de respaldo

ngrok 3.39.11 ya fue instalado desde Microsoft Store. Solo si preparas otra computadora, instala siguiendo las [instrucciones oficiales de Windows](https://ngrok.com/download/windows):

```powershell
winget install ngrok -s msstore
```

Abre una PowerShell nueva si el comando no aparece. Crea tu cuenta ngrok y configura su authtoken localmente.
Ese authtoken NO es QA_API_TOKEN; la configuración de ngrok lo guarda fuera del proyecto. Para no escribir
el valor literal en el historial de PowerShell:

```powershell
$ngrokSecret = Read-Host 'Authtoken de tu cuenta ngrok' -AsSecureString
$ngrokValue = [System.Net.NetworkCredential]::new('', $ngrokSecret).Password
ngrok config add-authtoken $ngrokValue
Remove-Variable ngrokSecret,ngrokValue
```

No uses transcripciones ni grabación de pantalla durante la configuración; el valor se pasa al proceso local.
Una vez autorizada la prueba pública, detén Cloudflare y abre el respaldo:

```powershell
ngrok http 8000
```

Repite el ensayo desde la otra red con la URL HTTPS de ngrok y una salida distinta:

```powershell
.\.venv\Scripts\python.exe -m scripts.rehearse --url https://TU-DOMINIO.ngrok-free.app --external-network --network-note "Segundo equipo con datos moviles; respaldo" --output logs/ensayo-ngrok-01.json
```

Se verificó `ngrok version`, pero falta la configuración de la cuenta y probar su túnel. `ngrok config check` indicó que todavía no existe ngrok.yml; configura el authtoken localmente como se explica arriba.

## 5. Detener de forma segura

1. Pulsa Ctrl+C en la ventana del túnel: deja de exponer la API.
2. Pulsa Ctrl+C en la ventana de uvicorn y espera el mensaje de apagado.
3. En esa misma ventana ejecuta `Remove-Item Env:QA_API_TOKEN -ErrorAction SilentlyContinue` y cierra las
   ventanas de ensayo. Borra también esa variable en cualquier otra ventana donde la hayas configurado.
4. No borres modelos ni checkpoints y no mates todos los procesos Python del equipo.

Para el ensayo oficial: coordina el horario, envía la URL activa al profesor, usa su token correspondiente,
conecta la laptop a corriente, evita suspensión y conserva API+túnel abiertos durante la sesión.
Todavía falta resolver/revisar el fallo público 04, la distribución autorizada de pesos y la defensa del equipo.
No crear `submission` hasta revisar todos los requisitos y recibir autorización final.
