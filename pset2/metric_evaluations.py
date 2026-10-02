import numpy as np
# Evaluation metrics based on the assignment guidelines
def metrics(posterior, actual_state):
    probabilities = posterior.values
    states = np.asarray(posterior.state_names["PL_disc"], dtype=float)

    # MAP prediction
    predicted_state = states[np.argmax(probabilities)]

    # Match Accuracy
    match = int(predicted_state == actual_state)

    # Distance Metric
    dm = abs(predicted_state - actual_state)

    # Expected State
    expected_state = np.sum(states * probabilities)

    # Distance Expectation Metric
    dem = abs(actual_state - expected_state)

    # Expected discrete distance
    dde = np.sum(probabilities * np.abs(states - actual_state))

    # NDDE
    ndde = dde / (len(states)-1)

    return{
        "Actual": actual_state,
        "Predicted": predicted_state,
        "Expected": expected_state,
        "MA": match,
        "DM": dm,
        "DEM": dem,
        "NDDE": ndde
    }

    