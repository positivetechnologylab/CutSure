from qiskit import QuantumCircuit, transpile, assemble, Aer, execute
from qiskit.providers.aer.noise import NoiseModel, errors, thermal_relaxation_error
from qiskit.visualization import plot_bloch_multivector, plot_histogram
from qiskit.quantum_info import Statevector
from qiskit.circuit.random import random_circuit
from qiskit_ibm_runtime import QiskitRuntimeService
import random
import warnings

# TOKEN = "bcc6639d56e2cae11f01bda3b35b3c287432a24710d11e805b45a5f9517e265a688779e956c386acd31720b0bc84c307db9a416eab049a20438c245c6db0ebbd"
# INSTANCE = None

"""
Note: in the state 0011, it is the top two wires (qubits 0 and 1) 
that are in the |1> state, and the bottom two wires (qubits 2 and 3) are in the |0> state.
"""

def generate_noise_model(qc, error_rate_1q=0.01, error_rate_2q=0.1, T1=None, T2=None, gate_time_1q=0.1e-6, gate_time_2q=0.2e-6):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
    one_qubit_gates = set()
    two_qubit_gates = set()

    one_qubit_gate_count = 0
    two_qubit_gate_count = 0

    for instr, _, _ in qc.data:
        if instr.num_qubits == 1:
            one_qubit_gates.add(instr.name)
            one_qubit_gate_count += 1
        elif instr.num_qubits == 2:
            two_qubit_gates.add(instr.name)
            two_qubit_gate_count += 1

    total_gates = one_qubit_gate_count + two_qubit_gate_count

    # Compute the weighted average gate time
    gate_time = (one_qubit_gate_count * gate_time_1q + two_qubit_gate_count * gate_time_2q) / total_gates

    gate_error_1q = errors.depolarizing_error(error_rate_1q, 1)
    gate_error_2q = errors.depolarizing_error(error_rate_2q, 2)

    noise_model = NoiseModel()

    # Add depolarizing errors
    for gate in one_qubit_gates:
        noise_model.add_all_qubit_quantum_error(gate_error_1q, gate)
    for gate in two_qubit_gates:
        noise_model.add_all_qubit_quantum_error(gate_error_2q, gate)

    # Add thermal relaxation errors if T1 and T2 are provided
    if T1 and T2:
        dec_error = thermal_relaxation_error(T1, T2, gate_time)
        for qubit in range(qc.num_qubits):
            for gate in one_qubit_gates:
                noise_model.add_quantum_error(dec_error, gate, [qubit])

    return noise_model

#qubit_range is a tuple of min,max values; same with depth_range; output is a list of circuits in qasm format
def generate_random_circuits(num_circuits, qubit_range, depth_range):
    #Generate random circuits using Qiskit's random_circuit. The random circuits have a random number of qubits defined in a range in qubit_range; same w/depth_range
    circuits = []
    for i in range(num_circuits):
        num_qubits = random.randint(qubit_range[0], qubit_range[1])
        depth = random.randint(depth_range[0], depth_range[1])
        circ = random_circuit(num_qubits, depth, measure=False, max_operands=2)
        circuits.append(circ)

    #Sort the circuits by number of qubits
    circuits.sort(key=lambda x: x.num_qubits)
    
    #print the circuits' qubit count

    for i in range(len(circuits)):
        print("Circuit",i,"has",circuits[i].num_qubits,"qubits")
    return circuits

def compile_to_na(base_circuit):
    num_qubits = base_circuit.num_qubits
    min_cz = float('inf')
    min_circuit = None
    for trial in range(3):
        circuit = transpile(base_circuit,
                            initial_layout=list(range(num_qubits)),
                            basis_gates=['u3', 'cz'],
                            optimization_level=3)
        if circuit.count_ops()['cz'] < min_cz:
            min_cz = circuit.count_ops()['cz']
            min_circuit = circuit

    return min_circuit

def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

def manhattan_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return abs(x2 - x1) + abs(y2 - y1)

#Returns new mapping of logical qubits in circuit to hardware qubits in np matrix
def map_qubits_to_region(matrix, circuit):
    num_qubits = circuit.num_qubits
    region = define_region(matrix, num_qubits)

    qubit_mapping = {qubit: position for qubit, position in zip(range(num_qubits), region)}

    return qubit_mapping, region

#defines the region of the hardware qubit matrix that the circuit will be run on
def define_region(matrix, desired_elements):
    rows, cols = matrix.shape
    center = (rows // 2, cols // 2)
    region = set([center])
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Down, Up, Right, Left

    while len(region) < desired_elements:
        for r, c in list(region): 
            if len(region) >= desired_elements:
                break
            for dr, dc in directions:
                new_r, new_c = r + dr, c + dc
                if 0 <= new_r < rows and 0 <= new_c < cols:
                    region.add((new_r, new_c))

    return region

#Fills the hardware circuit with qubits after circuit run's ejections
def fill_hardware_circuit(matrix, region):
    rows, cols = matrix.shape
    region_set = set(region)  # Convert list or tuple of region coordinates to a set for efficient lookup

    def find_nearest_one(matrix, zero_pos):
        nearest_dist = float('inf')
        nearest_one_pos = None

        for i in range(rows):
            for j in range(cols):
                # Check if the position is outside the region and has a 1
                if (i, j) not in region_set and matrix[i, j] == 1:
                    dist = euclidean_distance(zero_pos, (i, j))
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_one_pos = (i, j)

        return nearest_one_pos, nearest_dist

    # Iterate over the region
    move_count = 0
    move_dist = 0
    for pos in region:
        if matrix[pos] == 0:
            one_pos, dist = find_nearest_one(matrix, pos)
            if one_pos:
                matrix[one_pos] = 0  # "Move" the 1-element
                matrix[pos] = 1
                move_count += 1
                move_dist += dist
            else:
                matrix[:, :] = 1  # Not enough 1s outside, fill entire matrix
                return matrix, 0, 0, 1  # Return the matrix and a flag indicating that the atom array was reloaded

    return matrix, move_count, move_dist, 0  # Return the matrix,  and a flag indicating that the atom array was not reloaded

#Simulate execution of the circuit on the hardware qubit matrix by setting ejected qubits to empty
def simulate_circuit_on_matrix(matrix, counts, qubit_mapping, region):
    rows, cols = matrix.shape
    total_move_ct = 0
    total_move_dist = 0
    total_reloads = 0

    for outcome, shots in counts.items():
        for _ in range(shots):
            matrix_copy = matrix.copy()
            #print(outcome)
            # Eject qubits based on the outcome
            for qubit_index, state in enumerate(reversed(outcome)):
                if state == '1':
                    matrix_pos = qubit_mapping[qubit_index]
                    matrix_copy[matrix_pos[0]][matrix_pos[1]] = 0

            # Refill the matrix after each circuit run
            matrix_copy, move_ct, move_dist, reloads = fill_hardware_circuit(matrix_copy, region) 
            
            total_move_ct += move_ct
            total_move_dist += move_dist
            total_reloads += reloads

            if matrix_copy.shape != matrix.shape:
                print("Shape mismatch:", matrix_copy.shape, matrix.shape)
                continue  # or handle the mismatch as needed

            matrix[:] = matrix_copy
            #visualize_matrix_with_region(matrix,region)

    return matrix, total_move_ct, total_move_dist, total_reloads


def visualize_matrix_with_region(matrix, region):
    rows, cols = matrix.shape
    for r in range(rows):
        for c in range(cols):
            if (r, c) in region:
                # Element inside the region
                print('0' if matrix[r, c] == 0 else '1', end=' ')
            else:
                # Element outside the region
                print('-' if matrix[r, c] == 0 else '+', end=' ')
        print()  # New line at the end of each row

def circuit_run(qc, num_shots, hardware_mat, indices=None):
    """
    Runs a quantum circuit for a specified number of shots. If 'indices' is provided, adds Pauli X gates to 
    specified qubits, "flips back" the qubits in the 'indices' list after measurement, and compares the 
    number of |1> states before and after this flipping. If 'indices' is None, just identifies qubits measured 
    in the |1> state more than 50% of the time.

    :param qc: QuantumCircuit, the quantum circuit to be executed.
    :param num_shots: int, the number of shots to run the circuit.
    :param indices: list or None, a list of qubit indices to apply Pauli X gates before measurement, or None.
    :return: list, a list of qubit indices where the qubit is measured in the |1> state more than 50% of the time.
    """
    qc = qc.copy()  # Make a copy of the circuit to avoid altering the original circuit

    # Apply Pauli X gates to specified qubits if indices are provided
    if indices is not None:
        qc.remove_final_measurements()
        for index in indices:
            qc.x(index)
        qc.measure_all()

    #Define the region of the hardware qubit matrix that the circuit will be run on
    qubit_mapping, region = map_qubits_to_region(hardware_mat, qc)

    # Execute the circuit on the qasm simulator
    #simulator = Aer.get_backend('qasm_simulator')
    # result = execute(qc, simulator, shots=num_shots).result()
    # counts = result.get_counts(qc)
    if INSTANCE:
        service = QiskitRuntimeService(channel="ibm_quantum", instance=INSTANCE, token=TOKEN)
    else:
        service = QiskitRuntimeService(channel="ibm_quantum", token=TOKEN)
    backend = service.get_backend("ibmq_qasm_simulator")

    # get noise model
    # T1 = 4  # in seconds
    # T2 = 1.49  # in seconds
    # error_rate_1q = 0.000127 #based on U3 gate error
    # error_rate_2q = 0.0048 #based on CZ gate error
    # gate_time_1q = 2e-6  # 2 microseconds in seconds
    # gate_time_2q = 0.8e-6  # 0.8 microseconds in seconds
    # # Suppress warnings
    # warnings.filterwarnings("ignore")
    # with warnings.catch_warnings():
    #     warnings.filterwarnings('ignore', category=UserWarning)

    # noise_model = generate_noise_model(qc, error_rate_1q, error_rate_2q, T1, T2, gate_time_1q, gate_time_2q)
    # result = execute(qc,backend,noise_model=noise_model,shots=num_shots).result()

    result = execute(qc,backend,shots=num_shots).result()

    #Save jobID result to file
    # r_id = result.job_id
    # num_gates = sum(qc.count_ops().values())
    # with open('../job_ids/job_id_log_'+str(qc.num_qubits)+'_'+str(num_gates)+'.txt', 'a') as file:
    #     file.write('\n')
    #     file.write(str(r_id))
    
    counts = result.get_counts()
    #print("CTS:",counts)

    #Simulate execution on the hardware qubit matrix
    matrix, total_move_ct, total_move_dist, total_reloads = simulate_circuit_on_matrix(hardware_mat, counts, qubit_mapping, region)

    # Initialize variables
    total_ones_before_flip = 0
    # Perform flipping logic if indices are provided
    if indices == None:
        adjusted_counts = counts.copy()
    else:
        adjusted_counts = {}
        total_ones_before_flip = sum(count * state[::-1].count('1') for state, count in counts.items())
        # print("tobf",total_ones_before_flip)
        # print("Inds",indices)
        # Adjust counts by flipping the qubits specified in 'indices'
        for state, count in counts.items():
            flipped_state = list(state)
            for index in indices:
                flipped_state[-1 - index] = '0' if flipped_state[-1 - index] == '1' else '1'
            flipped_state = ''.join(flipped_state)
            #adjusted_counts[flipped_state] = adjusted_counts.get(flipped_state, 0) + count
            adjusted_counts[flipped_state] = count
        #print("ADJ:",adjusted_counts)

    # Calculate the proportion of |1> measurements for each qubit
    num_qubits = qc.num_qubits
    proportions = {}
    for state, count in adjusted_counts.items():
        for i in range(num_qubits):
            if i not in proportions:
                proportions[i] = 0
            proportions[i] += (state[::-1][i] == '1') * count
    # print("PR:",proportions)
    # print("aDJ:",adjusted_counts)
    # print("PROP: ",proportions,num_shots)
    count_ones = proportions.copy()
    for qubit in proportions:
        proportions[qubit] /= num_shots

    # Identify qubits measured in the |1> state more than 50% of the time
    qubits_over_50_percent = [qubit for qubit, proportion in proportions.items() if proportion > 0.5]
    # print("QBO50:",qubits_over_50_percent)

    # Output the comparison of |1> states before and after flipping back if indices were provided
    if indices is not None:
        total_ones_after_flip = sum(count * state[::-1].count('1') for state, count in adjusted_counts.items())
        return count_ones, total_ones_before_flip, total_ones_after_flip, matrix, total_move_ct, total_move_dist, total_reloads, adjusted_counts
    else:
        tot_ones = sum(count * state[::-1].count('1') for state, count in counts.items())
        return count_ones, tot_ones, tot_ones, matrix, total_move_ct, total_move_dist, total_reloads, counts


def get_true_probabilities(qc, num_shots, hardware_mat, token=TOKEN):
    """
    Computes the ideal scenario for the number of |1> states before and after flipping based on the true 
    probability distribution of the quantum circuit.

    :param qc: QuantumCircuit, the quantum circuit to be analyzed.
    :param num_shots: int, the number of shots to run the circuit for statistical measurement.
    :return: tuple, the total number of |1> states before and after flipping.
    """
    qc.remove_final_measurements()
    shot_ct = 1024
    qc.measure_all()
    # print(qc.qasm())
    #Run circuit with no noise on 10K shots
    if INSTANCE:
        service = QiskitRuntimeService(channel="ibm_quantum", instance=INSTANCE, token=TOKEN)
    else:
        service = QiskitRuntimeService(channel="ibm_quantum", token=TOKEN)
    backend = service.get_backend("ibmq_qasm_simulator")
    res = execute(qc,backend,shots=shot_ct).result()
    counts = res.get_counts()
    # print(counts)
    #print(counts)
    #Save jobID result to file
    # r_id = res.job_id
    # num_gates = sum(qc.count_ops().values())
    # with open('../job_ids/job_id_log_'+str(qc.num_qubits)+'_'+str(num_gates)+'.txt', 'a') as file:
    #     file.write('\n')
    #     file.write(str(r_id))

    # Compute probabilities of each qubit being in |1> state
    # Calculate the proportion of |1> measurements for each qubit
    total_ones_before_flip = 0
    adjusted_counts = counts.copy()  # Initialize with original counts
    
    #Make dict that computes probability of each output state occuring from counts
    prob_dict = {}
    for state, count in adjusted_counts.items():
        prob_dict[state] = count/shot_ct

    num_qubits = qc.num_qubits
    proportions = {}
    for state, count in adjusted_counts.items():
        for i in range(num_qubits):
            if i not in proportions:
                proportions[i] = 0
            proportions[i] += (state[::-1][i] == '1') * count

    count_ones = proportions.copy()
    for qubit in proportions:
        proportions[qubit] /= shot_ct

    # Identify qubits measured in the |1> state more than 50% of the time
    qubits_over_50_percent = [qubit for qubit, proportion in proportions.items() if proportion > 0.5]
    # print(proportions, qubits_over_50_percent, counts)

    # Copy the circuit for simulation and add Pauli X gates to the identified qubits
    
    temp_qc = qc.copy()
    temp_qc.remove_final_measurements()
    for index in qubits_over_50_percent:
        temp_qc.x(index)
    temp_qc.measure_all()
    # print(temp_qc.qasm())
    # Run the circuit on a qasm simulator with num_shots
    if INSTANCE:
        service = QiskitRuntimeService(channel="ibm_quantum", instance=INSTANCE, token=TOKEN)
    else:
        service = QiskitRuntimeService(channel="ibm_quantum", token=TOKEN)
    backend = service.get_backend("ibmq_qasm_simulator")

    # res = execute(temp_qc,backend,shots=num_shots).result()
    # print("ONE:",counts)
    counts = res.get_counts()
    # print("TWO:",counts)
    #Simulate on the hardware qubit matrix
    qubit_mapping, region = map_qubits_to_region(hardware_mat, qc)
    matrix, total_move_ct, total_move_dist, total_reloads = simulate_circuit_on_matrix(hardware_mat, counts, qubit_mapping, region)

    # Calculate total |1> states before flipping
    total_ones_before_flip = sum(count * state[::-1].count('1') for state, count in counts.items())

    # Adjust counts by flipping the qubits identified earlier
    adjusted_counts = {}
    for state, count in counts.items():
        flipped_state = list(state)
        for index in qubits_over_50_percent:
            flipped_state[-1 - index] = '0' if flipped_state[-1 - index] == '1' else '1'
        flipped_state = ''.join(flipped_state)
        if flipped_state in adjusted_counts:
            adjusted_counts[flipped_state] += count
        else:
            adjusted_counts[flipped_state] = count

    # Calculate total |1> states after flipping
    total_ones_after_flip = sum(count * state[::-1].count('1') for state, count in adjusted_counts.items())

    return prob_dict, total_ones_before_flip, total_ones_after_flip, proportions, matrix, total_move_ct, total_move_dist, total_reloads, counts, qubits_over_50_percent


def get_worst_case(qc, num_shots, hardware_mat, token, instance=None):
    # qc.measure_all()
    #Run circuit with no noise on 10K shots
    service = QiskitRuntimeService(channel="ibm_quantum", instance=instance, token=token)
    backend = service.get_backend("ibmq_qasm_simulator")
    res = execute(qc,backend,shots=num_shots).result()
    counts = res.get_counts()

    #Simulate on the hardware qubit matrix
    qubit_mapping, region = map_qubits_to_region(hardware_mat, qc)
    
    matrix, total_move_ct, total_move_dist, total_reloads = simulate_circuit_on_matrix(hardware_mat, counts, qubit_mapping, region)

    num_qubits = qc.num_qubits
    one_counts = {}
    for state, count in counts.items():
        for i in range(num_qubits):
            if i not in one_counts:
                one_counts[i] = 0
            one_counts[i] += (state[::-1][i] == '1') * count

    return matrix, total_move_ct, total_move_dist, total_reloads, counts, one_counts
