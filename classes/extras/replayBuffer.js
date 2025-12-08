/**
 * ReplayBuffer for experience replay in reinforcement learning.
 * Stores experiences up to a specified capacity and allows sampling of random batches.
 */
export const ReplayBuffer = class {
    /**
     * Creates an instance of ReplayBuffer.
     * @param {number} capacity - Maximum number of experiences to store.
     */
    constructor(capacity) {
        /**
         * Maximum capacity of the buffer.
         * @type {number}
         */
        this.capacity = capacity;

        /**
         * Buffer array to store experiences.
         * @type {Array}
         */
        this.buffer = [];

        /**
         * Pointer to the next position to insert experience.
         * @type {number}
         */
        this.position = 0;
    }

    /**
     * Adds a new experience to the buffer.
     * If the buffer is full, it overwrites the oldest experience.
     * @param {*} experience - The experience to add (can be any data structure).
     */
    add(experience) {
        if (this.buffer.length < this.capacity) {
            this.buffer.push(experience);
        } else {
            this.buffer[this.position] = experience;
            this.position = (this.position + 1) % this.capacity;
        }
    }

    /**
     * Samples a random batch of experiences from the buffer.
     * @param {number} batchSize - Number of experiences to sample.
     * @returns {Array} Array containing sampled experiences.
     */
    sample(batchSize) {
        const samples = [];
        for (let i = 0; i < batchSize; i++) {
            const idx = Math.floor(Math.random() * this.buffer.length);
            samples.push(this.buffer[idx]);
        }
        return samples;
    }
}