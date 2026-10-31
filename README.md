# Uber Fare Predictor — Machine Learning + FastAPI + Docker

Aplicación académica para estimar tarifas históricas de viajes de Uber en Nueva York. **No calcula tarifas reales actuales.**

## Dataset
Archivo `uber.csv` con 200,000 registros y variables `fare_amount`, `pickup_datetime`, coordenadas de origen/destino y `passenger_count`. Se excluyen filas con valores faltantes, coordenadas fuera de la región, tarifas no plausibles y distancias extremas. El CSV original **no se incluye en el repositorio** por tamaño; se conserva la referencia al archivo original de Kaggle y se reconstruye el modelo mediante `train.py`.

## Ejecutar localmente

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
mkdir data
# Copiar archivo original a data/uber.csv
python train.py --data data/uber.csv
pytest -q
uvicorn app.main:app --reload --port 8080
```

Visitar http://localhost:8080 y documentación de API http://localhost:8080/docs.

## Docker

El modelo ya entrenado se entrega en `model/uber_model.joblib`.

```bash
docker build -t uber-fare-predictor:1.1 .
docker run --rm -p 8080:8080 uber-fare-predictor:1.1
curl http://localhost:8080/health
```

O bien `docker compose up --build`.

## Endpoint

`POST /predict` con JSON:

```json
{"pickup_datetime":"2015-05-07T19:52:00Z","pickup_latitude":40.738354,"pickup_longitude":-73.999817,"dropoff_latitude":40.723217,"dropoff_longitude":-73.999512,"passenger_count":1}
```

Respuesta: `estimated_fare_usd`, `distance_km`, `note`. Datos inválidos devuelven HTTP 422. `GET /health` informa disponibilidad.

## Despliegue

Ver `docs/Manual_despliegue.docx` para Cloud Run y `docs/Validacion_pruebas.docx` para evidencias.
