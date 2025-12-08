/**
 * Policy Gradient Method (PGM)
 * A simple implementation of a policy gradient for a discrete action space.
 */

export const PolicyGradient = class {
    /**
     * Initializes the Policy Gradient agent.
     * @param {number} stateSize - Size of the state space.
     * @param {number} actionSize - Number of possible actions.
     * @param {number} learningRate - Learning rate for gradient ascent.
     */
    constructor(stateSize, actionSize, learningRate = 0.01) {
        this.stateSize = stateSize;
        this.actionSize = actionSize;
        this.learningRate = learningRate;

        // Initialize policy weights randomly
        this.weights = new Array(stateSize).fill(0).map(() => new Array(actionSize).fill(0));
    }

    /**
     * Softmax function to convert scores into probabilities.
     * @param {number[]} scores - Array of scores.
     * @returns {number[]} - Probabilities summing to 1.
     */
    softmax(scores) {
        const maxScore = Math.max(...scores);
        const expScores = scores.map(s => Math.exp(s - maxScore));
        const sumExp = expScores.reduce((a, b) => a + b, 0);
        return expScores.map(e => e / sumExp);
    }

    /**
     * Chooses an action based on current policy probabilities.
     * @param {number[]} state - Current state representation.
     * @returns {number} - Selected action index.
     */
    selectAction(state) {
        const scores = this.weights.reduce((acc, row, i) => {
        const score = row.reduce((sum, w, j) => sum + w * state[j], 0);
        acc.push(score);
        return acc;
        }, []);

        const probs = this.softmax(scores);
        // Sample action based on probabilities
        const rand = Math.random();
        let cumulative = 0;
        for (let i = 0; i < probs.length; i++) {
        cumulative += probs[i];
        if (rand < cumulative) {
            return i;
        }
        }
        return probs.length - 1; // fallback
    }

    /**
     * Updates policy weights using the collected episode data.
     * @param {Array} episodeStates - Array of states in the episode.
     * @param {Array} episodeActions - Array of actions taken.
     * @param {Array} episodeRewards - Array of rewards received.
     */
    update(episodeStates, episodeActions, episodeRewards) {
        // Calculate discounted rewards
        const discountedRewards = [];
        let cumulative = 0;
        for (let i = episodeRewards.length - 1; i >= 0; i--) {
        cumulative = episodeRewards[i] + 0.99 * cumulative;
        discountedRewards.unshift(cumulative);
        }

        // Normalize rewards
        const mean = discountedRewards.reduce((a, b) => a + b, 0) / discountedRewards.length;
        const std = Math.sqrt(discountedRewards.map(r => Math.pow(r - mean, 2)).reduce((a, b) => a + b, 0) / discountedRewards.length);
        const normalizedRewards = discountedRewards.map(r => (r - mean) / (std + 1e-8));

        // Gradient ascent step
        for (let t = 0; t < episodeStates.length; t++) {
        const state = episodeStates[t];
        const action = episodeActions[t];
        const reward = normalizedRewards[t];

        // Compute scores
        const scores = this.weights.reduce((acc, row, i) => {
            const score = row.reduce((sum, w, j) => sum + w * state[j], 0);
            acc.push(score);
            return acc;
        }, []);

        const probs = this.softmax(scores);

        // Update weights
        for (let i = 0; i < this.stateSize; i++) {
            for (let j = 0; j < this.actionSize; j++) {
            const indicator = (j === action) ? 1 : 0;
            const grad = (indicator - probs[j]) * state[i] * reward;
            this.weights[i][j] += this.learningRate * grad;
            }
        }
        }
    }
}