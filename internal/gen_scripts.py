import os
import sys
sys.path.append('./benchmark')

from torchbenchmark import _list_model_paths

DEST = 'models'

def main():
  with open('template.py') as f:
    temp = f.read()

  for x in _list_model_paths():
    name = x.split('/')[-1]
    res = temp.format(model_name=name)
    print(name)
    with open(os.path.join(DEST, name + '.py'), 'w') as f:
      f.write(res)
      f.write('\n')


if __name__ == '__main__':
  main()
    
