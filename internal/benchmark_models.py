import subprocess

from torchbenchmark import _list_model_paths

passed_model_list = []
failed_model_list = []

for model_path in _list_model_paths():
  print(" =============================================================================== ")
  name = model_path.split('/')[-1]
  print(" $$$ Testing model:", name)
  
  if name == 'cm3leon_generate':
    print("skip cm3leon_generate due to no output for a long time like 3 mins, skip it for now to get all test results")
    continue

  path_name = "models/" + name + ".py"
  return_code = subprocess.run(["python", path_name], capture_output=True, text=True) 

  if return_code.stderr:
    failed_model_list.append(name)
    print(f" $$$ Return code: {return_code.stderr}")
  else:
    passed_model_list.append(name)
    print(f" $$$ This model passed ! ")

  print(" =============================================================================== ")
