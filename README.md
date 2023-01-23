

# Install `pyenv`

Aν δεν υπαρχει η Python (που θελουμε να βαλουμε) στο συστυμα .
- παταμε δεξι click στο εικονιδιο των window
- windowPowerShell (admin)
- εγκαταστουμε το pyenv (https://github.com/pyenv/pyenv)
- Credentinals: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine
- `Install: Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"`

# Enviroment

- `pipenv --python 3.10`
- προσθετουμε τον Interpriter που μολις δημιουργηθηκε, απο κατω δεξια
- pipenv install