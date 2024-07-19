# NPLM-GWAK -- dictionary-learning edition
apply NPLM method to GWAK outputs using dictionary-learning strategy.

## Files description:
### Background-only toys:
- `run_reference.py`: file to submit bkg-only toys
  
  Args:
  - `--pythonscript` (`-p`, type: `str`): name of the python script to be launched
  - `--toys` (`-t`, type: `int`): number of toys to be run (they are sent as separate slurm jobs on Cannon)
  - `--local` (`-l`, type: `bool`, default: `False`): if run locally
  - `--firstseed` (`-s`, type: `int`): seed of first toy, the following have seeds created sequentially

  Example: run 10 bkg-only toys starting with seed=1
  ```
  python run_reference.py -t 10 -s 1 -p toy_reference.py
  ```
- `toy_reference.py`: script to run one bkg-only toy
### Signal-enriched toys:
- `run_sliding-window.py`: file to submit signal-enriched toys
  
  Args:
  - `--pythonscript` (`-p`, type: `str`): name of the python script to be launched
  - `--toys` (`-t`, type: `int`): number of toys to be run (they are sent as separate slurm jobs on Cannon)
  - `--local` (`-l`, type: `bool`, default: `False`): if run locally
  - `--slidingwindow` (`-s`, type: `int`): location of the slidingwindow analysed in the toy, the others are defined sequentially following the first one.
      
  Example: run 10 signal-enriched toys starting with slidingwindow=1
  ```
  python run_sliding-window.py -t 10 -s 1 -p toy_sliding-window.py
  ```
- `toy_sliding-window.py`: script to run one signal-enriched toy
