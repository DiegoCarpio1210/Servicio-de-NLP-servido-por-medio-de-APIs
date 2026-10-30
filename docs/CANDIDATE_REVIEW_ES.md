# Candidata quality40k: revisión antes de activarla

La aplicación habitual sigue utilizando `continued20k`, checkpoint-1600. El
dashboard conserva sus métricas históricas. No cambies únicamente `MODEL_DIR`:
la nueva candidata requiere tanto sus pesos como su política de extracción y
umbral, y todavía necesita aprobación para activarse en producción.

## Configuración congelada

- Pesos: `models/quality40k/model.safetensors`, exportados desde checkpoint-2000.
- SHA-256: `d42cc12e1ba17bc04e1ff652bd7e3bc7292d6d9ce39032501820c470b844d60f`.
- Tokenizer y configuración: archivos locales del mismo directorio; su identidad
  con el checkpoint está verificada en `logs/quality20261029/artifact_audit.json`.
- Política: máximo de 30 tokens; puntuar cada span como
  `start_logit + end_logit - 0.5 * log(longitud_en_tokens)`.
- Null: mínimo CLS entre ventanas; devolver null cuando
  `null_score - mejor_span > -0.40494251441686124`.
- Secuencia 384, stride 128, inferencia CPU, cuatro hilos.
- Fuente reproducible: perfil `trained` en `logs/quality20261029/final_lock.json`.

La política no contiene reglas por pregunta, entidades ni ejemplos públicos. Su
selección y calibración terminaron antes de puntuar la reserva final, con registro
en el commit `37fc668`. No deben modificarse con los resultados finales conocidos.

## Cómo se probó sin cambiar la aplicación

`experiments/candidate_api.py` utiliza la misma fábrica de API y autenticación
académicas con un adaptador de inferencia experimental. El verificador inicia una
única instancia local en `127.0.0.1:8000`, genera un token en memoria, comprueba el
hash y el umbral, ejecuta las pruebas y cierra sus procesos. No abre túneles.

Los resultados se guardan en `logs/quality20261029/cpu_*/`. No vuelvas a ejecutar
el verificador sobre evidencias existentes: se niega a sobrescribirlas. No hay
tokens guardados en estos resultados. Las predicciones completas de la evaluación
están excluidas de Git; las métricas, manifiestos de IDs y configuración sí se
publican.

## Después de la aprobación

La integración pendiente deberá activar conjuntamente pesos, política y umbral,
actualizar los hashes esperados por el lanzador y asociar el dashboard a las
métricas del modelo realmente activo. Después se repetirán las comprobaciones
críticas de la interfaz y API. Se conservará `continued20k` como respaldo.

Por ahora, para utilizar la versión estable, ejecuta `INICIAR_PIMIENTERO.bat` y
cierra con `DETENER_PIMIENTERO.bat`. La aprobación de la candidata no implica
autorizar un túnel público ni crear la etiqueta `submission`.
