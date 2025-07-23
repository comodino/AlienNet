# AlienNet: Riconoscimento Specie Aliene in Ambienti Acquatici

## 🚀 Avvio rapido


## ⚙️ Come Avviare l’Ambiente (Docker + Jupyter)
### Dal terminale:

```bash
docker-compose up --build
```

### Apre JupyterLab a:
http://localhost:8888

#### Da JupyterLab puoi eseguire training/train.py e tutti gli script di utilità tramite celle con:
```bash 
!python training/train.py
```

## ▶️ Esecuzione con Script Python
### Puoi eseguire direttamente il training usando uno script Python (main.py):
```bash 
docker-compose run aliennet python main.py
```
