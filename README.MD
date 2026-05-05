# PyBEAST++



PyBEAST++ is a refactoring of the PyBEAST courseware from the University of Leeds.

(Additional contributions from James Borgars and John Hodrien.)



Changes include bug fixes, vector standardisation to NumPy, framerate increases, an easier setup, and a move from WxWidgets to Qt for the GUI.





# Installation and running: 



**Option 1**



**(tested on School of Computer Science Linux Machines and Windows 11)**



Requirements: pre-installed anaconda environment



Running:



Within Pybeast directory

1. Open an anaconda shell

2\. First time running:



> conda env create -f environment.yaml

> conda activate pybeast

> python main.py



3\. Once your environment is created, on following uses, only run activate and main.

By default simulations use a fixed random seed (`12345`) so repeated runs are comparable. To change or disable this, pass `random_seed=<int>` or `random_seed=None` when calling `Simulation.__init__` in a simulation class.



**Option 2**



Provided is a bash script which utilises Apptainer (Singularity), that will create the image on the first run, and run the image (re-binding the drive such that all changes are saved) without recreating on further runs.

For more detailed instructions see PyBEAST apptainer GitHub repository.

