# Predictor de llegada de amigos

Prototipo de terminal que interpreta una quedada escrita en lenguaje natural y
predice el orden de llegada de sus asistentes. La IA, cuando está configurada,
solo convierte el mensaje en datos estructurados; un modelo de machine learning
entrenado con el histórico CSV es quien predice los minutos de retraso.

El entrenamiento compara un baseline por amigo con Random Forest y Extra Trees.
La separación train/test se hace por `quedada_id`, evitando que una misma
quedada aparezca en ambos conjuntos. El pipeline completo de scikit-learn se
guarda en `models/model.joblib`.

## Instalación

```bash
python -m venv .venv
```

En Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

En macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`OPENAI_API_KEY` es opcional. Si no existe, si la API falla o si devuelve JSON
inválido, se activa el parser local por reglas. `OPENAI_MODEL` y el nombre que
representa `yo` (`SELF_NAME`) se pueden editar en `.env`.

## Entrenar

```bash
python train.py
```

El comando localiza el CSV automáticamente, imprime su resumen y las métricas
MAE/RMSE/R², selecciona el modelo con menor MAE y guarda el artefacto. Si se
ejecuta la aplicación sin artefacto, lo entrena automáticamente.

## Ejecutar

```bash
python app.py
```

Mensaje principal de la demo:

```text
vamos a cenar a donde siempre y somos carlos, german, gaston y delgado
```

También se puede grabar una ejecución reproducible sin escribir en el prompt:

```bash
python app.py --message "vamos a cenar a donde siempre y somos carlos, german, gaston y delgado"
```

El modo técnico muestra el JSON interpretado, las observaciones construidas por
persona y las predicciones sin redondear:

```bash
python app.py --debug
```

## Pruebas

```bash
pytest -q
```

Las pruebas cubren el happy path, normalización de nombres y acentos, fechas y
horas inferidas, `todos`, `yo`, ranking real y separación por quedada sin
data leakage.

