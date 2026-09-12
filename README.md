# Customer Churn Intelligence

Production architecture:

- **Vercel** serves the static frontend in `frontend/`.
- **Railway** runs the FastAPI ML backend in `backend/`.
- The frontend sends CSV files to Railway through HTTP requests.

The production flow is:

```text
Browser -> Vercel frontend -> Railway FastAPI API -> Random Forest model
                                      -> predictions, SHAP, risk, recommendations
```

## Final project structure

```text
frontend/                 Vercel static frontend
  index.html
  style.css
  app.js
backend/                  Railway backend
  server.py
  ml_model.py
  requirements.txt
  models/churn_model.pkl
  data/demo_dataset.csv
.gitignore
README.md
```

There is one production copy of each frontend, backend, model, and dataset file. The old Streamlit surface and its duplicate root artifacts were removed after their useful model/business logic was migrated into `backend/`.

## Local testing

### Backend

```powershell
cd backend
pip install -r requirements.txt
$env:FRONTEND_URL = "http://localhost:5500"
uvicorn server:app --reload
```

The backend runs at `http://localhost:8000`. The training script expects the clean dataset at `backend/data/demo_dataset.csv` and saves `backend/models/churn_model.pkl`:

```powershell
python ml_model.py
```

API endpoints:

- `GET /health`
- `POST /predict/file` with a multipart CSV field named `file`
- `POST /predict` with `{ "rows": [...] }`
- `GET /feature-importance`
- `GET /docs`

### Frontend

From the repository root, start a static server:

```powershell
python -m http.server 5500 --directory frontend
```

Open `http://localhost:5500`. The single frontend API configuration is the `API_BASE_URL` constant at the top of `frontend/app.js`. It defaults to `http://localhost:8000`; replace that one value with the Railway public HTTPS URL before the Vercel deployment.

## Manual GitHub commands

Run these commands yourself when the project is ready:

```powershell
git status
git add .
git commit -m "Finalize Vercel frontend and Railway backend"
git push -u origin main
```

No deployment command is run by this project setup.

## Manual Railway deployment

1. Open Railway.
2. Create a new project.
3. Deploy from the GitHub repository.
4. Select this repository.
5. Set the **Root Directory** to `backend`.
6. Set the **Start Command** to:

   ```text
   uvicorn server:app --host 0.0.0.0 --port $PORT
   ```

7. Add `FRONTEND_URL` with the exact Vercel origin, such as `https://your-project.vercel.app`.
8. Deploy manually.
9. Copy the generated Railway public URL.
10. Verify `https://your-railway-url/health` returns a successful response.

Railway runs only `backend/`. It uses the dynamic `$PORT` value supplied by Railway.

## Manual Vercel deployment

1. Open Vercel.
2. Import the same GitHub repository.
3. Set the **Root Directory** to `frontend`.
4. Deploy the static frontend with no build command.
5. In `frontend/app.js`, replace `API_BASE_URL` with the Railway backend URL.
6. Commit that URL change and redeploy if required.

Vercel serves only `frontend/`; it does not run Python, FastAPI, or Streamlit.

## CORS and security

FastAPI reads the comma-separated `FRONTEND_URL` environment variable and allows only those configured origins. Use the exact production Vercel origin rather than a wildcard. Local development defaults to `http://localhost:5500`.
