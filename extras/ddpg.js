import { NeuralNetwork } from "./neuralNetwork.js";
import { ReplayBuffer } from "./replayBuffer.js";
/**
 * Deep Deterministic Policy Gradient (DDPG) Agent for continuous action spaces.
 */
export const DDPGAgent = class {
    /**
     * Creates an instance of DDPGAgent.
     * @param {number} stateSize - Dimension of the state space.
     * @param {number} actionSize - Dimension of the action space.
     * @param {Object} [options={}] - Optional parameters for agent configuration.
     * @param {number} [options.gamma=0.99] - Discount factor for future rewards.
     * @param {number} [options.actorLearningRate=0.001] - Learning rate for the actor network.
     * @param {number} [options.criticLearningRate=0.002] - Learning rate for the critic network.
     * @param {number} [options.memoryCapacity=10000] - Capacity of the experience replay buffer.
     * @param {number} [options.batchSize=64] - Batch size for training.
     */
    constructor(stateSize, actionSize, options = {}) {
        this.stateSize = stateSize;
        this.actionSize = actionSize;
        this.gamma = options.gamma ?? 0.99;
        this.actorLearningRate = options.actorLearningRate ?? 0.001;
        this.criticLearningRate = options.criticLearningRate ?? 0.002;
        this.memory = new ReplayBuffer(options.memoryCapacity ?? 10000);
        this.batchSize = options.batchSize ?? 64;
        // Actor network
        this.actor = new NeuralNetwork(
            [this.stateSize],      // (constructor signature in your file allows different forms;
                                // we'll fully init via train(...) below)
            this.actionSize,
            {
            hiddenLayers: [64, 64],
            learningRate: this.actorLearningRate,
            activation: 'ReLU',
            type: 'regression',
            debug: false,
            }
        );
        // Critic network
        this.critic = new NeuralNetwork(
            [this.stateSize + this.actionSize],
            1,
            {
            hiddenLayers: [64, 64],
            learningRate: this.criticLearningRate,
            activation: 'ReLU',
            type: 'regression',
            debug: false,
            }
        );

        // 🔹 Initialize networks with proper features arrays and dummy samples
        this._initNetworks();
    }

    
    /** Initialize NN shapes/weights using an empty training pass with proper features arrays. */
    _initNetworks() {
        // Build features arrays as numeric indices (what your NN expects)
        this.actorFeatures  = Array.from({ length: this.stateSize },              (_, i) => i);
        this.criticFeatures = Array.from({ length: this.stateSize + this.actionSize }, (_, i) => i);

        // Dummy samples so NeuralNetwork._prepare/_initNetwork can infer shapes
        const actorDummy  = Object.fromEntries(this.actorFeatures.map(i => [i, 0]));
        actorDummy.label  = 0; // regression scalar target

        const criticDummy = Object.fromEntries(this.criticFeatures.map(i => [i, 0]));
        criticDummy.label = 0; // regression scalar target

        // epoch: 0 => only initialize, no training
        this.actor.train([actorDummy],  this.actorFeatures,  { epoch: 0, batch: 1, type: 'regression' });
        this.critic.train([criticDummy], this.criticFeatures, { epoch: 0, batch: 1, type: 'regression' });
    }



    /**
     * Selects an action for a given state, with optional exploration noise.
     * @param {Array<number>} state - Current state.
     * @param {number} [noiseStd=0.1] - Standard deviation of Gaussian noise added for exploration.
     * @returns {Array<number>} - Action vector.
     */
    act(state, noiseStd = 0.1) {
        const out = this.actor.predict(state); // returns array (e.g., [y])
        const a = Array.isArray(out) ? Number(out[0]) : Number(out);
        const noisy = a + noiseStd * (Math.random() * 2 - 1);
        return [ Math.max(-1, Math.min(1, noisy)) ];
    }

    /**
     * Stores an experience tuple in the replay buffer.
     * @param {Array<number>} state - Current state.
     * @param {Array<number>} action - Action taken.
     * @param {number} reward - Reward received.
     * @param {Array<number>} nextState - Next state after action.
     * @param {boolean} done - Whether the episode has terminated.
     */
    remember(state, action, reward, nextState, done) {
        this.memory.add({ state, action, reward, nextState, done });
    }

    /**
     * Performs training over a batch of experiences.
     * Uses the critic to estimate Q-values and updates both networks.
     */
    
    train() {
        if (this.memory.buffer.length < this.batchSize) return;
        const batch = this.memory.sample(this.batchSize);

        for (const experience of batch) {
            const { state, action, reward, nextState, done } = experience;

            
            // --- Critic target ---
            const nextAction = this.actor.predict(nextState);
            const qNextArr   = this.critic.predict(nextState.concat(nextAction));
            const qNextVal   = Array.isArray(qNextArr) ? qNextArr[0] : qNextArr;
            const qTarget    = done ? reward : reward + this.gamma * qNextVal;

            // --- Critic update: (s,a) -> qTarget ---
            const criticInput = state.concat(action);

            // Prefer single-sample helper if present
            if (typeof this.critic.trainOne === 'function') {
            this.critic.trainOne(criticInput, qTarget);
            } else {
            // 🔹 Fallback: build object sample + features array
            const criticSample = {};
            for (let i = 0; i < criticInput.length; i++) criticSample[i] = criticInput[i];
            criticSample.label = qTarget;

            this.critic.train([criticSample], this.criticFeatures, { batch: 1, epoch: 1, type: 'regression' });
            }

            // --- Actor update via numerical gradient ---
            const currentAction = this.actor.predict(state);  // returns [a]
            const delta = 1e-3;
            const grad = currentAction.map((a, i) => {
            const plusA  = currentAction.map((v, j) => (j === i ? v + delta : v));
            const minusA = currentAction.map((v, j) => (j === i ? v - delta : v));

            const plusQArr  = this.critic.predict(state.concat(plusA));
            const minusQArr = this.critic.predict(state.concat(minusA));

            const plusQ  = Array.isArray(plusQArr)  ? plusQArr[0]  : plusQArr;
            const minusQ = Array.isArray(minusQArr) ? minusQArr[0] : minusQArr;

            return (plusQ - minusQ) / (2 * delta);
            });

            const alpha = this.actorLearningRate;
            const improvedAction = currentAction.map((a, i) => a + alpha * grad[i]);
            const clippedAction  = improvedAction.map(a => Math.max(-1, Math.min(1, a)));
            const actorSample = {};
            for (let i = 0; i < state.length; i++) actorSample[i] = state[i];
            actorSample.label = Array.isArray(clippedAction) ? Number(clippedAction[0]) : Number(clippedAction);
            this.actor.train([actorSample], this.actorFeatures, { batch: 1, epoch: 1, type: 'regression' });
        }
    }
};