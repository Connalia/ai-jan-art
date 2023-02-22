# Setup Enviroment

How to set up your environment with the libraries that need:

1) Run to command line: `pipenv --python 3.10`
2) Add the new Interpreter, from the bottom right in Pycharm
3) Run to command line: `pipenv install` or (`PIPENV_DONT_LOAD_ENV=1 pipenv install`)

**Note**

If there is not exist the specific Python version (which we want to install) in your PC.

Follow the instructions how to ''**install `pyenv`**'' on Windows:

- right click on the windows icon on windows bar
- select 'windowPowerShell (admin)'
- Run to command line: `install pyenv` (https://github.com/pyenv/pyenv)
- Run to command line: `Credentinals: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine`
- Run to command
  line: `Install: Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"`
