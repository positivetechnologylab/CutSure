from qiskit import QuantumCircuit
from quantum_module import circuit_run, get_true_probabilities, generate_random_circuits, compile_to_na
from qiskit.visualization import plot_bloch_multivector, plot_histogram
import os
import re
import sys
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
import warnings
from qiskit_ibm_runtime import QiskitRuntimeService
from datetime import datetime

# IMPORTANT NOTE: These were used to access IBM's quantum backend for its simulators before they were taken offline. This is now deprecated, and thus the code no longer functions.
# TOKEN = ""
# INSTANCE = None

def load_qasm_file(qasm_file_path):
    #Load a .qasm file and save its contents to a list of lists
    qasm_file = open(qasm_file_path, "r")
    qasm_contents = qasm_file.readlines()
    qasm_contents = [line.strip() for line in qasm_contents]
    qasm_file.close()
    return qasm_contents

def constant_increment(**kwargs):
    """ Constant increment strategy. """
    total_shots = kwargs.get('total_shots') 
    const = kwargs.get('const', 1)
    return int(const)

def proportional_increment(**kwargs):
    """ Increments as a proportion of total_shots. """
    total_shots = kwargs.get('total_shots')  
    proportion = kwargs.get('proportion', 0.1)
    return int(total_shots * proportion)

def linear_increment(current_step, **kwargs):
    """ Increments increase linearly, in proportion to total_shots. """
    total_shots = kwargs.get('total_shots') 
    proportion = kwargs.get('proportion', 0.1)
    const = kwargs.get('const', 0)
    return int(total_shots * proportion * current_step + const)

def quadratic_increment(current_step, **kwargs):
    """ Increments increase quadratically, in proportion to total shots. """
    total_shots = kwargs.get('total_shots')
    proportion = kwargs.get('proportion', 0.1)
    const = kwargs.get('const', 0)
    return int(total_shots * proportion * current_step**2 + const)

def generate_increasing_list(**kwargs):
    """ Generates a list of increasing integers based on the increment strategy. """
    total_shots = kwargs.get('total_shots')
    increment_strategy = kwargs.get('increment_strategy')

    if increment_strategy==constant_increment or increment_strategy==proportional_increment:
        increment = increment_strategy(**kwargs)
        values = [increment] * int(total_shots/increment)
        if sum(values) < total_shots:
            values.append(total_shots - sum(values))
        # print("const",values)
        return values
    else:
        values = []
        current_value = 0
        current_step = 0
        while current_value < total_shots:
            current_step += 1
            increment = increment_strategy(current_step,**kwargs)
            current_value += increment
            if sum(values) + increment <= total_shots:
                values.append(increment)
            else:
                values.append(total_shots - sum(values))
                return values

# The way this works is that the increment is based on how much the probability changed in the last iteration
def entropic_increment(previous_proportions, current_proportions, total_shots, shots_done, **kwargs):
    base_const = kwargs.get('base_const', 10)  # Initial base increment
    sensitivity = kwargs.get('sensitivity', 0.02)  # Sensitivity to changes

    entropy = sum(abs(current_proportions.get(qubit, 0) - previous_proportions.get(qubit, 0)) for qubit in current_proportions)

    dynamic_const = base_const + int(shots_done / total_shots * base_const)

    if entropy > 0:
        increment = min(dynamic_const + int((1 / entropy) * sensitivity * total_shots), total_shots - shots_done)
    else:
        increment = total_shots - shots_done  # If no entropy, use remaining shots

    return increment

class FilterWarningsStream:
    def write(self, text):
        if not text.startswith("WARNING"):
            sys.__stdout__.write(text)

    def flush(self):
        sys.__stdout__.flush()

def main():
    warnings.filterwarnings("ignore")
    sys.stdout = FilterWarningsStream()
    #generate random circuits' qasms
    qubit_size_range = (16,28)
    depth_range = (4,15)
    num_circuits = 3
    tot_shots = 1024
    QiskitRuntimeService.save_account(channel="ibm_quantum", token=TOKEN, set_as_default=True, overwrite=True)
    #GENERATE RANDOM CIRCUITS
    print(TOKEN)
    # qasm_circs = generate_random_circuits(num_circuits, qubit_size_range, depth_range)

    start_time = time.time()

    
    #...OR LOAD CIRCUITS FROM QASM FILES
    # For each qasm file in ../specific_circuits/ , load the qasm file and append it to qasm_circs
    qasm_circs = []
    direc = "../specific_circuits"+"5"+"/"
    for filename in os.listdir(direc):
        qasm_contents = load_qasm_file(direc+filename)
        print(filename)
        qasm_circs.append(QuantumCircuit.from_qasm_str("\n".join(qasm_contents)))

    #Use numpy to create a 16x16 matrix that represents qubits. Each circuit is mapped to the "center" of the matrix.
    hardware_mat = np.ones((16, 16))

    #Amount of ejection reduction {filename: avg_ejection_reduction}
    eject_reduction = {}
    eject_reduction_ideal = {}
    results = {}
    all_det_results = {}

    circ_id = 0
    # Iterate over each file

    # Suppress specific Qiskit warnings
    warnings.filterwarnings("ignore", category=UserWarning, message="Specific error for instruction.*")

    for og_circ in qasm_circs:
        sum_flip_diff = 0
        print("Running circuit: with qubit size",og_circ.num_qubits,"and depth",og_circ.depth())
        file = "../charts/"+str(og_circ.num_qubits)+'_'+str(og_circ.depth())+"_ideal_counts.png"

        #Convert circuit to u3/cz basis
        #circ = compile_to_na(og_circ)
        circ = og_circ

        # Load QASM file contents
        qasm_contents = circ.qasm()

        #save qasm_contents to a temporary file called "tmp.qasm"
        tmp_file = open("tmp.qasm","w")
        tmp_file.write(qasm_contents)
        tmp_file.close()

        det_res_dict = {}
        det_res_dict['ideal'] = {}
        det_res_dict['circ'] = circ
        #Generate ideal probabilities for each output state for the quantum circuit (ex: {'000': 0.5, '001': 0.4, ...})

        ideal_counts, ideal_qubit_1_pre_counts, ideal_qubit_1_post_counts, ideal_prob, matrix, total_move_ct, total_move_dist, total_reloads, counts, qubits_over_50_percent = get_true_probabilities(circ, tot_shots, hardware_mat)
        det_res_dict['ideal'][tot_shots] = (ideal_counts, ideal_qubit_1_pre_counts, ideal_qubit_1_post_counts, ideal_prob, matrix, total_move_ct, total_move_dist, total_reloads, counts, qubits_over_50_percent)

        #Generate a list of numbers of shots to run in a group
        inc_lists = []
        inc_lists.append((constant_increment,generate_increasing_list(total_shots=tot_shots,increment_strategy=constant_increment,const=101)))
        inc_lists.append((proportional_increment,generate_increasing_list(total_shots=tot_shots,increment_strategy=proportional_increment,proportion=0.3)))
        inc_lists.append((linear_increment,generate_increasing_list(total_shots=tot_shots,increment_strategy=linear_increment,proportion=0.05,const=0)))
        inc_lists.append((quadratic_increment,generate_increasing_list(total_shots=tot_shots,increment_strategy=quadratic_increment,proportion=0.05,const=0)))
        inc_lists.append((entropic_increment,None))

        res_dict = {'no_policy':ideal_qubit_1_pre_counts,constant_increment:None,proportional_increment:None,linear_increment:None,quadratic_increment:None,entropic_increment:None,'ideal':ideal_qubit_1_post_counts}
        inds = []
        for strat, increment_list in inc_lists:
            print(strat)
            det_res_dict[str(strat)] = {}
            # Generate probabilities in model with no noise
            shots_ran = 0
            counts = {}
            indices = None
            total_pre_flip_1_counts = 0 #total number of times any qubit is in the |1> state before reverting qubits to "natural" state
            total_post_flip_1_counts = 0 #total number of times any qubit is in the |1> state after reverting qubits to "natural" state (represents 'worst' case)
            ct = 0
            if strat != entropic_increment:
                for num_shots in increment_list:
                    if counts == {}:
                        counts, total_pre_flip_1_counts, total_post_flip_1_counts, hardware_mat, total_move_ct, total_move_dist, total_reloads, adjusted_counts = circuit_run(circ,num_shots,hardware_mat)
                        det_res_dict[str(strat)][(num_shots,ct)] = (counts, total_pre_flip_1_counts, total_post_flip_1_counts, hardware_mat, total_move_ct, total_move_dist, total_reloads, adjusted_counts, [])
                        # print("NEWCT:",counts, str(num_shots))
                        # print(indices)
                    else:
                        new_counts, i, j, hardware_mat, total_move_ct, total_move_dist, total_reloads, adjusted_counts = circuit_run(circ, num_shots, hardware_mat, indices)
                        det_res_dict[str(strat)][(num_shots,ct)] = (new_counts, i, j, hardware_mat, total_move_ct, total_move_dist, total_reloads, adjusted_counts, indices)
                        total_pre_flip_1_counts += i
                        total_post_flip_1_counts += j
                        # print("NEWCT:",new_counts, str(num_shots))
                        for qubit, count in new_counts.items():
                            counts[qubit] = counts.get(qubit, 0) + count
                        # print("ADJCT:",counts, str(num_shots))
                    ct += 1
                    shots_ran += num_shots
                    # Calculate the proportion of |1> measurements for each qubit
                    proportions = {qubit: count / shots_ran for qubit, count in counts.items()}
                    # Update indices for qubits measured in the |1> state more than 50% of the time
                    indices = [qubit for qubit, proportion in proportions.items() if proportion > 0.5]
                    inds.append(indices)
                    # print("INDS:",inds)
                    
            else:
                previous_proportions = {}
                proportions = {}  # Initialize proportions
                const = 10
                sensitivity = 0.02
                while shots_ran < tot_shots:
                    # Determine the number of shots for this iteration
                    if shots_ran == 0:
                        num_shots = const  # Start with the base increment
                    else:
                        # Calculate increment based on the change in proportions
                        num_shots = entropic_increment(previous_proportions, proportions, tot_shots, shots_ran, const=const, sensitivity=sensitivity)
                    
                    if counts == {}:
                        counts, total_pre_flip_1_counts, total_post_flip_1_counts, hardware_mat, total_move_ct, total_move_dist, total_reloads, adjusted_counts = circuit_run(circ, num_shots,hardware_mat)
                        det_res_dict[str(strat)][(num_shots,ct)] = (counts, total_pre_flip_1_counts, total_post_flip_1_counts, hardware_mat, total_move_ct, total_move_dist, total_reloads, adjusted_counts, [])
                    else:
                        new_counts, i, j, hardware_mat, total_move_ct, total_move_dist, total_reloads, adjusted_counts = circuit_run(circ, num_shots, hardware_mat, indices)
                        det_res_dict[str(strat)][(num_shots,ct)] = (new_counts, i, j, hardware_mat, total_move_ct, total_move_dist, total_reloads, adjusted_counts, indices)
                        total_pre_flip_1_counts += i
                        total_post_flip_1_counts += j
                        for qubit, count in new_counts.items():
                            counts[qubit] = counts.get(qubit, 0) + count
                    ct += 1
                    shots_ran += num_shots
                    #print(num_shots,shots_ran)
                    # Update previous proportions for the next iteration
                    previous_proportions = proportions.copy()

                    # Calculate the proportion of |1> measurements for each qubit
                    proportions = {qubit: count / shots_ran for qubit, count in counts.items()}
                    # Update indices for qubits measured in the |1> state more than 50% of the time
                    indices = [qubit for qubit, proportion in proportions.items() if proportion > 0.5]

                    # Break the loop if the total shots limit is reached
                    if shots_ran >= tot_shots:
                        break
            
            res_dict[strat] = (total_pre_flip_1_counts, total_post_flip_1_counts)
        # Initialize variables to find min and max values
        min_value = float('inf')
        max_value = float('-inf')
        #print(res_dict)
        # Iterate over the dictionary items to find min and max values
        for key, value in res_dict.items():
            if isinstance(value, tuple):
                first_val, second_val = value
                min_value = min(min_value, first_val)
                max_value = max(max_value, second_val)

        # Set 'ideal' and 'no_policy' appropriately
        res_dict['ideal'] = min(res_dict.get('ideal', float('inf')), min_value)
        res_dict['no_policy'] = max(res_dict.get('no_policy', float('-inf')), max_value)
        # Update the values to be the first element of the tuple for non-string keys
        print(res_dict)
        for key in list(res_dict.keys()):  # Use list to create a copy of keys
            if key not in ['ideal', 'no_policy']:
                res_dict[key] = res_dict[key][0]
        # Change the keys to be strings corresponding to the increment strategy
        res_dict['constant_increment'] = res_dict.pop(constant_increment)
        res_dict['proportional_increment'] = res_dict.pop(proportional_increment)
        res_dict['linear_increment'] = res_dict.pop(linear_increment)
        res_dict['quadratic_increment'] = res_dict.pop(quadratic_increment)
        res_dict['entropic_increment'] = res_dict.pop(entropic_increment)
        print(res_dict)
        # Append the modified res_dict to results
        results[circ_id] = res_dict
        all_det_results[circ_id] = det_res_dict
        circ_id += 1
    time.sleep(1)
    # End time
    end_time = time.time()

    # Calculate runtime
    runtime = end_time - start_time

    # Convert to hours, minutes, and seconds
    hours = int(runtime // 3600)
    minutes = int((runtime % 3600) // 60)
    seconds = int(runtime % 60)

    # Format as a string
    timestr = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    print(timestr)

    # Extracting hours, minutes, seconds, and microseconds

    tokenstr = TOKEN[:5]
    time
    #Save results to a pkl file that is timestamped in the file name
    # file_name = "../results/"+str(qubit_size_range[0])+"_"+str(qubit_size_range[1])+"_"+str(depth_range[0])+"_"+str(depth_range[1])+"_"+str(num_circuits)+"_"+str(tot_shots)+".pkl"
    file_name = '../results/'+tokenstr+'_'+timestr+'.pkl'
    file = open(file_name,"wb")
    pickle.dump(results,file)
    file.close()

    # file_name = "../results/"+str(qubit_size_range[0])+"_"+str(qubit_size_range[1])+"_"+str(depth_range[0])+"_"+str(depth_range[1])+"_"+str(num_circuits)+"_"+str(tot_shots)+"_det.pkl"
    file_name = '../det_results/'+tokenstr+'_'+timestr+'_det.pkl'
    file = open(file_name,"wb")
    pickle.dump(all_det_results,file)
    file.close()

    print("RUNITME:",runtime)

warnings.filterwarnings('ignore', category=UserWarning)

if __name__ == "__main__":
    main()
