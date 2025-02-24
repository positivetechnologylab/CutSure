# CutSure: Increasing the Efficiency of Neutral Atoms by Reducing Qubit Waste from Measurement-related Ejections

***This work won the Graduate category in the [2024 ACM SC Student Research Competition (SRC)](https://src.acm.org/winners/2025), and will be proceeding to the SRC Grand Finals.*** Access the abstract [**`here`**](CutSure_Extended_Abstract.pdf) and the poster [**`here`**](CutSure Poster-2.pdf).

## Important Note: This project is no longer functional due to IBM taking their backend simulation offline. This code is provided for reference purposes.

## Usage
1. In your terminal, run `git clone https://github.com/positivetechnologylab/CutSure.git` to clone the repository.
2. Run `cd CutSure`.
3. Create a virtual environment if necessary, and run `pip install -r requirements.txt` to install the requirements.
4. Run the notebook `main.py`, which contains various functions that are used to produce the output ejection related data.
    - The noise model can be uncommented out of the 'quantum_module.py' file for noisy runs.

## Side Effects
Upon completion, 'main.py' will produce a dictionary that contains all of the relevant output information. The results will be saved to a corresponding directory of the same name.

## Repository Structure
- [**`main.py`**](main.py): The script containing the core code for producing the results of the different ejection schemes, such as entropic or quadratic.
- [**`quantum_module.py`**](quantum_module): The script that contains a lot of the ancillary code for executing CutSure. This includes the optional noise model and the Qiskit code for actually executing circuits.
- [**`README.md`**](README.md): Repository readme with setup and execution instructions.
- [**`requirements.txt`**](requirements.txt): Requirements to be installed before running the Python scripts.

## Copyright
Copyright © 2025 Positive Technology Lab. All rights reserved. For permissions, contact ptl@rice.edu.
