import { MLMath } from "../classes/mlmath.class.js";

/**
 * Class implementing the DBSCAN clustering algorithm.
 */
export const DBScan = class{
    /**
     * Creates an instance of DBScan.
     * @param {number} eps - The maximum distance for two points to be considered neighbors.
     * @param {number} minPts - Minimum number of points required to form a dense region (cluster).
     */
    constructor(eps, minPts) {
        this.eps = eps;
        this.minPts = minPts;
        this.labels = [];
    }

    /**
     * Finds all points within eps radius of the given point.
     * @param {number[][]} points - Array of all points.
     * @param {number} pointIdx - Index of the point to query.
     * @returns {number[]} - Array of indices of neighboring points.
     */
    regionQuery(points, pointIdx) {
        const neighbors = [];
        const point = points[pointIdx];

        for (let i = 0; i < points.length; i++) {
        if (MLMath.distance(point, points[i],'euclidean') <= this.eps) {
            neighbors.push(i);
        }
        }
        return neighbors;
    }

    /**
     * Expands the cluster from a seed point.
     * @param {number[][]} points - Array of all points.
     * @param {number[]} labels - Array of cluster labels for points.
     * @param {number} pointIdx - Index of the seed point.
     * @param {number} clusterId - Current cluster ID.
     */
    expandCluster(points, labels, pointIdx, clusterId) {
        const seeds = this.regionQuery(points, pointIdx);
        for (let i = 0; i < seeds.length; i++) {
        labels[seeds[i]] = clusterId;
        }

        let i = 0;
        while (i < seeds.length) {
        const currentP = seeds[i];
        const neighbors = this.regionQuery(points, currentP);
        if (neighbors.length >= this.minPts) {
            for (let j = 0; j < neighbors.length; j++) {
            if (labels[neighbors[j]] === undefined || labels[neighbors[j]] === -1) {
                if (labels[neighbors[j]] !== clusterId) {
                labels[neighbors[j]] = clusterId;
                seeds.push(neighbors[j]);
                }
            }
            }
        }
        i++;
        }
    }

    /**
     * Runs the DBSCAN clustering algorithm on the provided points.
     * @param {number[][]} points - Array of data points to cluster.
     * @returns {number[]} Array of cluster labels for each point. Noise points are labeled as -1.
     */
    fit(points) {
        this.labels = new Array(points.length);
        let clusterId = 0;

        for (let i = 0; i < points.length; i++) {
        if (this.labels[i] !== undefined) continue; // Already processed

        const neighbors = this.regionQuery(points, i);

        if (neighbors.length < this.minPts) {
            this.labels[i] = -1; // Mark as noise
        } else {
            this.labels[i] = clusterId;
            this.expandCluster(points, this.labels, i, clusterId);
            clusterId++;
        }
        }
        return this.labels;
    }
}