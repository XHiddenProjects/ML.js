export const ReplayMemory = class{
    /**
     * Creates a replay memory buffer
     * @param {number} capacity - Maximum number of experiences to store
     */
    constructor(capacity) {
        this.capacity = capacity;
        this.memory = [];
        this.position = 0;
    }

    /**
     * Adds a new experience to the memory
     * @param {Object} experience - Experience object containing state, action, reward, nextState, done
     */
    push(experience) {
        if (this.memory.length < this.capacity) {
        this.memory.push(experience);
        } else {
        this.memory[this.position] = experience;
        }
        this.position = (this.position + 1) % this.capacity;
    }
    
    /**
     * Returns the size of the memory
     *
     * @returns {Number} Memory size
     */
    size(){
        return this.memory.length;
    }

    /**
     * Samples a batch of experiences
     * @param {number} batchSize - Number of experiences to sample
     * @returns {Object[]} - Array of experience objects
     */
    sample(batchSize) {
        const samples = [];
        const len = this.memory.length;
        for (let i = 0; i < batchSize; i++) {
        const index = Math.floor(Math.random() * len);
        samples.push(this.memory[index]);
        }
        return samples;
    }

    /**
     * Checks if the memory has enough samples for a batch
     * @param {number} batchSize
     * @returns {boolean}
     */
    canSample(batchSize) {
        return this.memory.length >= batchSize;
    }
}