echo symlinks appear to work following this procedure. 1) mklink/J 2) git add 3) append sys.path 4) import

mklink/J mlBridge ..\mlBridge
mklink/J acbllib ..\acbllib
mklink/J chatlib ..\chatlib
mklink/J streamlitlib ..\streamlitlib

git add mlBridge\mlBridgeLib.py
git add acbllib
git add chatlib\chatlib.py
git add streamlitlib\streamlitlib.py

echo change python file to add all paths. e.g. sys.path.append(str(pathlib.Path.cwd().joinpath('mlBridgeLib'))) # global
echo afterwards import libs. e.g. import mlBridgeLib

