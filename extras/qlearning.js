/**
 * Class representing a Q-Learning agent.
 */
export const QLearning = class{
    /**
     * Creates a QLearning instance.
     * @param {number} numStates - The number of states in the environment.
     * @param {number} numActions - The number of possible actions.
     * @param {number} learningRate - The learning rate (alpha).
     * @param {number} discountFactor - The discount factor (gamma).
     * @param {number} epsilon - The exploration rate.
     */
    constructor(numStates, numActions, learningRate, discountFactor, epsilon) {
        this.numStates = numStates;
        this.numActions = numActions;
        this.learningRate = learningRate;
        this.discountFactor = discountFactor;
        this.epsilon = epsilon;

        // Initialize Q-table with zeros
        this.qTable = Array.from({ length: numStates }, () =>
        Array.from({ length: numActions }, () => 0)
        );
    }

    /**
     * Selects an action based on the current state using an epsilon-greedy policy.
     * @param {number} state - The current state.
     * @returns {number} - The selected action.
     */
    chooseAction(state) {
        if (Math.random() < this.epsilon) {
        // Exploration: random action
        return Math.floor(Math.random() * this.numActions);
        } else {
        // Exploitation: best known action
        const actions = this.qTable[state];
        const maxQ = Math.max(...actions);
        // In case of multiple actions with same max Q-value, pick randomly among them
        const maxActions = actions
            .map((q, index) => ({ q, index }))
            .filter(item => item.q === maxQ)
            .map(item => item.index);
        return maxActions[Math.floor(Math.random() * maxActions.length)];
        }
    }

    /**
     * Updates the Q-table based on the agent's experience.
     * @param {number} state - The previous state.
     * @param {number} action - The action taken.
     * @param {number} reward - The reward received.
     * @param {number} nextState - The next state after taking the action.
     */
    updateQTable(state, action, reward, nextState) {
        const currentQ = this.qTable[state][action];
        const maxNextQ = Math.max(...this.qTable[nextState]);
        const newQ =
        currentQ +
        this.learningRate * (reward + this.discountFactor * maxNextQ - currentQ);
        this.qTable[state][action] = newQ;
    }

    /**
     * Gets the Q-table.
     * @returns {number[][]} - The current Q-table.
     */
    getQTable() {
        return this.qTable;
    }
}