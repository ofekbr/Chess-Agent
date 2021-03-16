import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def nnTrain(clock_path, move_num_path, threats_path, moves_available_path, taken_path, materials_path, times_path):
    input_threats = np.loadtxt(threats_path, dtype=int).tolist()
    input_moves_available = np.loadtxt(moves_available_path, dtype=int).tolist()
    input_clock = np.loadtxt(clock_path, dtype=int).tolist()
    input_taken = np.loadtxt(taken_path).tolist()
    input_move_num = np.loadtxt(move_num_path, dtype=int).tolist()
    input_materials = np.loadtxt(materials_path, dtype=int).tolist()
    # input_enemy_times=np.loadtxt(enemy_time_path, dtype=int).tolist()
    input_times = np.loadtxt(times_path, dtype=int)
    temp_input_times = np.zeros((len(input_times)), dtype=int)
    top_first_class = 4
    top_second_class = 20
    top_third_class = 100
    zeros = 0
    ones = 0
    twos = 0
    threes = 0
    for i in range(len(input_times)):
        if input_times[i] <= top_first_class:
            temp_input_times[i] = 0
            zeros += 1
        elif input_times[i] <= top_second_class:
            temp_input_times[i] = 1
            ones += 1
        elif input_times[i] <= top_third_class:
            temp_input_times[i] = 2
            twos += 1
        else:
            temp_input_times[i] = 3
            threes += 1

    print(f'zeros = {zeros}    ones = {ones}     twos = {twos}      threes = {threes}')
    print(
        f'zeros = {zeros / len(input_threats)}    ones = {ones / len(input_threats)}      twos = {twos / len(input_threats)}     threes = {threes / len(input_threats)}')
    input_times = temp_input_times

    threat_array = np.array(input_threats)
    moves_available_array = np.array(input_moves_available)
    clock_array = np.array(input_clock)
    move_num_array = np.array(input_move_num)
    taken_array = np.array(input_taken)
    input_materials = np.array(input_materials)
    input_times = np.array(input_times)
    # input_enemy_times = np.array(input_enemy_times)
    # complete_array = np.vstack(
    #     (threat_array, moves_available_array, clock_array, taken_array, move_num_array, input_materials,
    #      input_enemy_times,input_times)).transpose()
    complete_array = np.vstack(
        (clock_array, move_num_array, threat_array, moves_available_array, taken_array, input_materials,
         input_times)).transpose()
    complete_array = complete_array.astype(float)

    dataset = complete_array
    X = dataset[:, 0:6]
    Y = dataset[:, 6]
    # split data into train and test sets
    seed = 7
    test_size = 0.1
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)  #
    # fit model no training data
    model = XGBClassifier()
    weights = []
    for i in y_train:
        if i == 0:
            weights.append(3.164)
        elif i == 1:
            weights.append(2.518)
        elif i == 2:
            weights.append(4.016)
        else:
            weights.append(28.011)

    model.fit(X_train, y_train, sample_weight=weights)
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    for i, j in zip(predictions, y_test):
        print(f'prediction = {int(i)}   label = {int(j)}')
    print("Accuracy: %.2f%%" % (accuracy * 100.0))


nnTrain('atleast2400s_400diff_above2000elo/masters_clock.txt',
        'atleast2400s_400diff_above2000elo/masters_moves.txt',
        'atleast2400s_400diff_above2000elo/masters_threats.txt',
        'atleast2400s_400diff_above2000elo/masters_available_moves.txt',
        'atleast2400s_400diff_above2000elo/masters_taken.txt',
        'atleast2400s_400diff_above2000elo/masters_materials.txt',
        'atleast2400s_400diff_above2000elo/masters_times.txt')
