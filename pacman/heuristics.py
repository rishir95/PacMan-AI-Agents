# heuristic for selecting the node
def scoreEvaluation(state):
    return state.getScore() + [0,-1000.0][state.isLose()] + [0,1000.0][state.isWin()]

def normalizedScoreEvaluation(rootState, currentState):
    rootEval = scoreEvaluation(rootState);
    currentEval = scoreEvaluation(currentState);
    return (currentEval - rootEval) / 1000.0;

# heuristic (remaining cost) for A*
def admissibleHeuristic(state):
    if state.isLose():
        return 1000.0;
    return state.getNumFood() + len(state.getCapsules());
