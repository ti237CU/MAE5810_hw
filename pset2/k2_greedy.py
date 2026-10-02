from pgmpy.estimators import K2
## K2 greedy search structure function
def k2_learn_structure(data, node_order, max_parents):
    score = K2(data)
    parents = {
        node: []
        for node in node_order
    }

    for i, node in enumerate(node_order):
        possible_parents = node_order[:i]   # Only allow earlier variables to be parents

        current_parents = []
        current_score = score.local_score(node, current_parents)

        while(len(current_parents) < max_parents):
            best_parent = None
            best_score = current_score

            for candidate in possible_parents:
                if candidate in current_parents:
                    continue

                candidate_parents = (current_parents + [candidate])
                candidate_score = (score.local_score(node, candidate_parents))

                if candidate_score > best_score:
                    best_score = candidate_score
                    best_parent = candidate

            if best_parent is None:
                break

            current_parents.append(best_parent)
            current_score = best_score

        parents[node] = current_parents
    return parents

