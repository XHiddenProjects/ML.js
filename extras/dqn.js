import { NeuralNetwork } from "./neuralNetwork.js";
import { ReplayMemory } from "./replayMemory.js";

export const DQNAgent = class {
    /**
     * Deep Q-Network Agent
     * @param {Object} config - Configuration object
     * @param {number} config.stateSize - Size of state vector
     * @param {number} config.actionSize - Number of possible actions
     * @param {number} config.hiddenSize - Number of neurons in hidden layer
     * @param {number} config.memoryCapacity - Replay memory capacity
     * @param {number} config.batchSize - Batch size for training
     * @param {number} config.gamma - Discount factor
     * @param {number} config.epsilon - Exploration rate
     * @param {number} config.epsilonDecay - Decay rate of epsilon
     * @param {number} config.epsilonMin - Minimum epsilon
     * @param {number} config.learningRate - Learning rate for training
     */
    constructor(config) {
        this.stateSize = config.stateSize;
        this.actionSize = config.actionSize;
        this.hiddenSize = config.hiddenSize || 24;
        this.memory = new ReplayMemory(config.memoryCapacity || 10000);
        this.batchSize = config.batchSize || 32;
        this.gamma = config.gamma || 0.95;
        this.epsilon = config.epsilon || 1.0;
        this.epsilonDecay = config.epsilonDecay || 0.995;
        this.epsilonMin = config.epsilonMin || 0.01;
        this.learningRate = config.learningRate || 0.01;
        // Initialize neural network
        this.model = new NeuralNetwork(this.stateSize, this.hiddenSize, this.actionSize);
        // Initialize reward tracking
        this._totalReward = 0;
        this._lastImmediateReward = 0;
    }

    /**
     * Selects an action based on epsilon-greedy policy
     * @param {number[]} state - Current state
     * @returns {number} - Action index
     */
    act(state) {
        if (Math.random() < this.epsilon) {
            return Math.floor(Math.random() * this.actionSize);
        } else {
            const qValues = this.model._forward(state);
            const flattenedActivations = qValues.activations.flat();
            return flattenedActivations.indexOf(Math.max(...flattenedActivations));
        }
    }

    /**
     * Stores experience in replay memory
     * @param {number[]} state
     * @param {number} action
     * @param {number} reward
     * @param {number[]} nextState
     * @param {boolean} done
     */
    remember(state, action, reward, nextState, done) {
        this.memory.push({ state, action, reward, nextState, done });
        // Update immediate reward
        this._lastImmediateReward = reward;
        // Update total reward
        this._totalReward += reward;
    }

    /**
     * Trains the neural network using a batch of experiences
     */
    replay() {
        if (!this.memory.canSample(this.batchSize)) return;
        const batch = this.memory.sample(this.batchSize);
        for (const experience of batch) {
            const { state, action, reward, nextState, done } = experience;
            const qValuesResult = this.model._forward(state);
            const targetQ = Array.from(qValuesResult.activations || qValuesResult);

            if (done) {
                targetQ[action] = reward;
            } else {
                const nextQResult = this.model._forward(nextState);
                const nextQ = Array.from(nextQResult.activations || nextQResult);
                targetQ[action] = reward + this.gamma * Math.max(...nextQ);
            }

            // Train the network to fit targetQ
            this.model.train(state, targetQ, {learningRate: this.learningRate});
        }

        // Decay epsilon
        if (this.epsilon > this.epsilonMin) {
            this.epsilon *= this.epsilonDecay;
        }
    }

    /**
     * Returns the total accumulated reward score
     * @returns {number}
     */
    rewardScore() {
        return this._totalReward;
    }

    /**
     * Returns the immediate reward received from the last action
     * @returns {number}
     */
    immediateReward() {
        return this._lastImmediateReward;
    }
}