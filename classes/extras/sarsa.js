/**
 * SARSA Reinforcement Learning Agent
 */
export const SARSA = class {
    /**
     * Creates an instance of SARSA.
     * @param {Array<string>} states - List of all possible states.
     * @param {Array<string>} actions - List of all possible actions.
     * @param {Object} [params] - Optional parameters.
     * @param {number} [params.alpha=0.1] - Learning rate.
     * @param {number} [params.gamma=0.9] - Discount factor.
     * @param {number} [params.epsilon=0.1] - Exploration rate.
     */
    constructor(states, actions, {
        alpha = 0.1,
        gamma = 0.9,
        epsilon = 0.1
    } = {}) {
        this.states = states;
        this.actions = actions;
        this.alpha = alpha;
        this.gamma = gamma;
        this.epsilon = epsilon;

        // Initialize Q-table with zeros
        this.Q = {};
        for (let state of states) {
        this.Q[state] = {};
        for (let action of actions) {
            this.Q[state][action] = 0;
        }
        }
    }

    /**
     * Selects an action based on epsilon-greedy policy.
     * @param {string} state - Current state.
     * @returns {string} - Selected action.
     */
    chooseAction(state) {
        if (Math.random() < this.epsilon) {
        // Explore: random action
        const randomIndex = Math.floor(Math.random() * this.actions.length);
        return this.actions[randomIndex];
        } else {
        // Exploit: best action
        const qValues = this.Q[state];
        let maxQ = -Infinity;
        let bestActions = [];
        for (let action of this.actions) {
            if (qValues[action] > maxQ) {
            maxQ = qValues[action];
            bestActions = [action];
            } else if (qValues[action] === maxQ) {
            bestActions.push(action);
            }
        }
        // If multiple actions have the same Q-value, pick one at random
        const randomIndex = Math.floor(Math.random() * bestActions.length);
        return bestActions[randomIndex];
        }
    }

    /**
     * Updates the Q-table using the SARSA update rule.
     * @param {string} state - Current state.
     * @param {string} action - Action taken.
     * @param {number} reward - Reward received.
     * @param {string} nextState - Next state after action.
     * @param {string} nextAction - Next action to be taken.
     */
    update(state, action, reward, nextState, nextAction) {
        const predict = this.Q[state][action];
        const target = reward + this.gamma * this.Q[nextState][nextAction];
        this.Q[state][action] += this.alpha * (target - predict);
    }

    /**
     * Runs a single episode of interaction with the environment.
     * @param {Object} env - Environment with reset() and step() methods.
     * @param {function} env.reset - Resets environment and returns initial state.
     * @param {function} env.step - Takes (state, action) and returns { nextState, reward, done }.
     * @param {number} [maxSteps=1000] - Maximum steps per episode.
     */
    runEpisode(env, maxSteps = 1000) {
        let state = env.reset();
        let action = this.chooseAction(state);
        for (let step = 0; step < maxSteps; step++) {
        const { nextState, reward, done } = env.step(state, action);
        const nextAction = this.chooseAction(nextState);
        this.update(state, action, reward, nextState, nextAction);
        state = nextState;
        action = nextAction;
        if (done) break;
        }
    }
}