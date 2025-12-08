import { SpellChecker } from "../extras/spell_checker.js";

export const NLP = class {
	constructor() {
			const language = navigator.language.split('-');
			// Multilingual support: store per-language models
			this.languages = new Set();
			this.models = {}; // lang -> model object (documents, answers, vocabulary, tokenCounts, docCount, totalDocs, totalTokensPerIntent, isTrained)
			this.activeLanguage = 'default';
			this.smoothing = 1; // Laplace smoothing
			this.spellChecker = new SpellChecker(language[0],language[1]);
			// create default language model
			this.addLanguage(this.activeLanguage);
	}

		addLanguage(lang) {
			if (!lang) throw new Error('language required');
			if (!this.languages.has(lang)) {
				this.languages.add(lang);
				this.models[lang] = {
					documents: [], // { text, tokens, intent }
					answers: {}, // intent -> [answers]
					vocabulary: new Set(),
					tokenCounts: {}, // intent -> { token: count }
					docCount: {}, // intent -> number
					totalDocs: 0,
					totalTokensPerIntent: {},
					isTrained: false,
				};
			}
			this.activeLanguage = lang;
			return this.models[lang];
		}

		_getModel(lang) {
            lang = lang||navigator.language.substring(0,2)||'en';
			const l = lang || this.activeLanguage;
			const m = this.models[l];
			if (!m) throw new Error(`language not found: ${l}`);
			return m;
		}
	tokenize(text) {
		if (!text) return [];
		const tokens = String(text)
			.toLowerCase()
			.split(/\W+/)
			.filter((t) => t.length > 1)
			.map((token) => {
			const corrected = this.spellChecker.spellCheckFromURL(token);
			return corrected === token ? token : corrected.split(' ')[0].toLowerCase();
			});
		return tokens
	}
	addDocument(text, intent) {
		const tokens = this.tokenize(text);
		// language-aware: optional third param 'lang'
		const _lang = arguments.length >= 3 ? arguments[0] : undefined;
		const model = this._getModel(_lang);
		model.documents.push({ text, tokens, intent });
		model.isTrained = false;
	}

	addAnswer(intent, answer) {
		const _lang = arguments.length >= 3 ? arguments[0] : undefined;
		const model = this._getModel(_lang);
		if (!model.answers[intent]) model.answers[intent] = [];
		model.answers[intent].push(answer);
	}

	train() {
		// Train model for a specific language (or active language if not provided)
		const _lang = arguments.length >= 1 ? arguments[0] : undefined;
		const model = this._getModel(_lang);
		// Reset counts
		model.vocabulary = new Set();
		model.tokenCounts = {};
		model.docCount = {};
		model.totalTokensPerIntent = {};
		model.totalDocs = model.documents.length;

		for (const doc of model.documents) {
			const intent = doc.intent;
			if (!model.tokenCounts[intent]) model.tokenCounts[intent] = {};
			if (!model.docCount[intent]) model.docCount[intent] = 0;
			if (!model.totalTokensPerIntent[intent]) model.totalTokensPerIntent[intent] = 0;

			model.docCount[intent]++;

			for (const token of doc.tokens) {
				model.vocabulary.add(token);
				model.tokenCounts[intent][token] = (model.tokenCounts[intent][token] || 0) + 1;
				model.totalTokensPerIntent[intent]++;
			}
		}

		model.isTrained = true;
	}

	_scoreForIntent(tokens, intent) {
		// Returns log-probability score for intent
		const model = this._getModel(arguments.length >= 2 ? arguments[2] : undefined);
		const docCountForIntent = (model.docCount && model.docCount[intent]) || 0;
		if (docCountForIntent === 0) return -Infinity;

		const prior = docCountForIntent / Math.max(1, model.totalDocs || 1);
		let logProb = Math.log(prior);

		const tokenCounts = model.tokenCounts[intent] || {};
		const totalTokens = model.totalTokensPerIntent[intent] || 0;
		const V = Math.max(1, model.vocabulary.size);

		for (const token of tokens) {
			const count = tokenCounts[token] || 0;
			const prob = (count + this.smoothing) / (totalTokens + this.smoothing * V);
			logProb += Math.log(prob);
		}
		return logProb;
	}

	classify(text) {
		// classify in a given language or active language
		const _lang = arguments.length >= 2 ? arguments[1] : undefined;
		const model = this._getModel(_lang);
		if (!model.isTrained) this.train(_lang);
		const tokens = this.tokenize(text);

		const intents = Object.keys(model.docCount);
		if (intents.length === 0) return { intent: null, score: 0 };

		const scores = {};
		let maxLog = -Infinity;
		for (const intent of intents) {
			const s = this._scoreForIntent(tokens, intent, _lang);
			scores[intent] = s;
			if (s > maxLog) maxLog = s;
		}

		// Convert log-scores to normalized probabilities (softmax) for readability
		const expScores = {};
		let sum = 0;
		for (const intent of intents) {
			// subtract maxLog for numerical stability
			const v = Math.exp(scores[intent] - maxLog);
			expScores[intent] = v;
			sum += v;
		}

		const probs = {};
		for (const intent of intents) probs[intent] = expScores[intent] / sum;

		// pick best
		let best = intents[0];
		for (const intent of intents) {
			if (probs[intent] > probs[best]) best = intent;
		}
		return { intent: best, score: probs[best], probabilities: probs };
	}

	process(text) {
		const _lang = arguments.length >= 2 ? arguments[0] : undefined;
        const txt = arguments.length >= 2 ? arguments[1] : text;
		const tokens = this.tokenize(txt);
		const classification = this.classify(txt, _lang);
		const intentName = classification.intent;
		const model = this._getModel(_lang);
		console.log(model);
		const answers = intentName ? (model.answers[_lang] ? model.answers[_lang][intentName] : model.answers[intentName]) : [];
		return {
			src: txt,
			tokens,
			intent: { name: intentName, score: classification.score },
			answers,
			classification: classification.probabilities,
		};
	}

	getAnswers(intent) {
		const _lang = arguments.length >= 2 ? arguments[1] : undefined;
        console.log(intent);
		const model = this._getModel(_lang);
        console.log(model);
		return model.answers[intent] ? Array.from(model.answers[intent]) : [];
	}

	toJSON() {
		// serialize all language models
		const models = {};
		for (const lang of this.languages) {
			const m = this.models[lang];
			models[lang] = {
				documents: m.documents,
				answers: m.answers,
				vocabulary: Array.from(m.vocabulary),
				tokenCounts: m.tokenCounts,
				docCount: m.docCount,
				totalDocs: m.totalDocs,
				totalTokensPerIntent: m.totalTokensPerIntent,
				isTrained: !!m.isTrained,
			};
		}
		return {
			activeLanguage: this.activeLanguage,
			models,
			smoothing: this.smoothing,
		};
	}

	static fromJSON(data) {
		const n = new NLP();
		if (data && data.models) {
			for (const lang of Object.keys(data.models)) {
				n.addLanguage(lang);
				const msrc = data.models[lang];
				const m = n.models[lang];
				m.documents = msrc.documents || [];
				m.answers = msrc.answers || {};
				m.vocabulary = new Set(msrc.vocabulary || []);
				m.tokenCounts = msrc.tokenCounts || {};
				m.docCount = msrc.docCount || {};
				m.totalDocs = msrc.totalDocs || m.documents.length;
				m.totalTokensPerIntent = msrc.totalTokensPerIntent || {};
				m.isTrained = !!msrc.isTrained;
			}
		}
		if (data && data.activeLanguage) n.activeLanguage = data.activeLanguage;
		n.smoothing = typeof data.smoothing === 'number' ? data.smoothing : 1;
		return n;
	}
};


/**
 * Compares 2 different images or videos and computes a similarity score.
 *
 * Features:
 *  - Supports Image/Canvas/ImageBitmap, Blob/File/ArrayBuffer, and URL(string) sources.
 *  - Video comparison by sampling frames across duration.
 *  - Computes multiple metrics: MSE, PSNR, SSIM, Histogram Intersection, pHash.
 *  - Returns a weighted aggregate similarity in [0..1] + per-metric details.
 *
 * NOTE: Designed for browser environments (uses <canvas>). If you use Node,
 *       provide a canvas polyfill or adapt loading functions.
 */
export const PixelDetection = class {
	/**
	 * @param {Object} [options]
	 * @param {number} [options.targetSize=256] - Square target dimension (px) for analysis.
	 * @param {'contain'|'cover'} [options.resizeMode='contain'] - Canvas fitting strategy.
	 * @param {boolean} [options.grayscale=true] - Convert to grayscale before metrics.
	 * @param {Object} [options.weights] - Metric weights; keys: mse, psnr, ssim, hist, phash.
	 * @param {number} [options.videoFrameSamples=12] - Number of frames to sample when comparing videos.
	 * @param {number} [options.maxPSNR=60] - Upper bound to normalize PSNR (dB) into [0..1].
	 */
	constructor(options = {}) {
		this.targetSize = options.targetSize ?? 256;
		this.resizeMode = options.resizeMode ?? 'contain';
		this.grayscale = options.grayscale ?? true;
		this.videoFrameSamples = options.videoFrameSamples ?? 12;
		this.maxPSNR = options.maxPSNR ?? 60;
		this.weights = Object.assign(
		{ mse: 1, psnr: 1, ssim: 2, hist: 1, phash: 2 },
		options.weights || {}
		);

		// Reusable offscreen canvas
		this._canvas = document.createElement('canvas');
		this._ctx = this._canvas.getContext('2d', { willReadFrequently: true });
	}

	/**
	 * Top-level compare function. Detects image vs video and routes accordingly.
	 * @param {HTMLImageElement|HTMLVideoElement|CanvasImageSource|Blob|File|ArrayBuffer|URL|String} a - Source A (Image/Video element, CanvasImageSource, Blob/File, ArrayBuffer, or URL string).
	 * @param {HTMLImageElement|HTMLVideoElement|CanvasImageSource|Blob|File|ArrayBuffer|URL|String} b - Source B (same accepted types).
	 * @returns {Promise<{similarity:number, metrics:Object, kind:'image'|'video'}>}
	 */
	async compare(a, b) {
		const aKind = await this._detectKind(a);
		const bKind = await this._detectKind(b);

		if (aKind !== bKind) {
		// If kinds differ (e.g., image vs video), we’ll try to compare first video frames to the image.
		// This provides a practical fallback rather than hard failing.
		if (aKind === 'video' && bKind === 'image') {
			return this._compareVideoToImage(a, b);
		} else if (aKind === 'image' && bKind === 'video') {
			const res = await this._compareVideoToImage(b, a);
			// swap perspective (b vs a)
			return res;
		}
		throw new Error('Unsupported: sources are of different kinds and no fallback applied');
		}

		return aKind === 'video'
		? this._compareVideos(a, b)
		: this._compareImages(a, b);
	}

	/* ----------------------------- Loading Helpers ---------------------------- */

	async _detectKind(src) {
		const el = await this._materializeSource(src);
		if (el instanceof HTMLVideoElement) return 'video';
		return 'image';
	}

	async _materializeSource(src) {
		// Already an element?
		if (src instanceof HTMLImageElement ||
			src instanceof HTMLCanvasElement ||
			typeof ImageBitmap !== 'undefined' && src instanceof ImageBitmap) {
		return src;
		}
		if (src instanceof HTMLVideoElement) return src;

		// Blob/File/ArrayBuffer/URL string
		if (src instanceof Blob) {
		if (src.type.startsWith('video/')) return await this._loadVideoFromBlob(src);
		return await this._loadImageFromBlob(src);
		}
		if (src instanceof ArrayBuffer) {
		const blob = new Blob([src]);
		return await this._materializeSource(blob);
		}
		if (typeof src === 'string') {
		// Heuristic by extension; safer to attempt image first then video
		const lower = src.toLowerCase();
		if (lower.match(/\.(mp4|webm|ogg|mov)$/)) {
			return await this._loadVideoFromURL(src);
		}
		return await this._loadImageFromURL(src);
		}

		throw new Error('Unsupported source type');
	}

	async _loadImageFromURL(url) {
		const img = new Image();
		img.crossOrigin = 'anonymous';
		img.decoding = 'async';
		img.src = url;
		await img.decode().catch(() => new Promise((res, rej) => {
		img.onload = () => res();
		img.onerror = (e) => rej(e);
		}));
		return img;
	}

	async _loadImageFromBlob(blob) {
		const url = URL.createObjectURL(blob);
		try {
		const img = await this._loadImageFromURL(url);
		return img;
		} finally {
		URL.revokeObjectURL(url);
		}
	}

	async _loadVideoFromURL(url) {
		const video = document.createElement('video');
		video.crossOrigin = 'anonymous';
		video.src = url;
		video.muted = true; // allows autoplay seeking on some browsers
		await this._awaitVideoReady(video);
		return video;
	}

	async _loadVideoFromBlob(blob) {
		const url = URL.createObjectURL(blob);
		try {
		const video = await this._loadVideoFromURL(url);
		return video;
		} finally {
		URL.revokeObjectURL(url);
		}
	}

	_awaitVideoReady(video) {
		return new Promise((resolve, reject) => {
		const onReady = () => {
			cleanup();
			resolve();
		};
		const onError = (e) => {
			cleanup();
			reject(e);
		};
		const cleanup = () => {
			video.removeEventListener('loadeddata', onReady);
			video.removeEventListener('error', onError);
		};
		if (video.readyState >= 2) return resolve();
		video.addEventListener('loadeddata', onReady);
		video.addEventListener('error', onError);
		});
	}

	/* ------------------------------ Canvas/Resize ----------------------------- */

	_drawToCanvas(source) {
		// Determine intrinsic size
		let sw, sh;
		if (source instanceof HTMLImageElement || source instanceof HTMLVideoElement) {
		sw = source.videoWidth || source.naturalWidth || source.width;
		sh = source.videoHeight || source.naturalHeight || source.height;
		} else if (source instanceof HTMLCanvasElement) {
		sw = source.width; sh = source.height;
		} else if (typeof ImageBitmap !== 'undefined' && source instanceof ImageBitmap) {
		sw = source.width; sh = source.height;
		} else {
		throw new Error('Unsupported source for canvas draw');
		}

		// Target canvas size
		const tw = this.targetSize, th = this.targetSize;
		this._canvas.width = tw;
		this._canvas.height = th;

		// Compute fit rect
		const srcAspect = sw / sh;
		const dstAspect = tw / th;
		let dw, dh;
		if (this.resizeMode === 'cover'
			? srcAspect < dstAspect
			: srcAspect > dstAspect) {
		// match width
		dw = tw;
		dh = tw / srcAspect;
		} else {
		// match height
		dh = th;
		dw = th * srcAspect;
		}
		const dx = (tw - dw) / 2;
		const dy = (th - dh) / 2;

		// Clear & draw
		this._ctx.clearRect(0, 0, tw, th);
		this._ctx.drawImage(source, dx, dy, dw, dh);

		return this._ctx.getImageData(0, 0, tw, th);
	}

	_toGrayscale(imageData) {
		const { data, width, height } = imageData;
		const gray = new Float32Array(width * height);
		for (let i = 0, p = 0; i < data.length; i += 4, p++) {
		const r = data[i], g = data[i+1], b = data[i+2];
		// Rec. 709 luminance
		gray[p] = 0.2126*r + 0.7152*g + 0.0722*b;
		}
		return { gray, width, height };
	}

	/* -------------------------------- Metrics -------------------------------- */

	_mse(grayA, grayB) {
		let mse = 0;
		for (let i = 0; i < grayA.length; i++) {
		const d = grayA[i] - grayB[i];
		mse += d * d;
		}
		mse /= grayA.length;
		// Normalize to [0..1] by max possible (255^2)
		const nmse = mse / (255 * 255);
		return { mse, nmse, similarity: Math.max(0, 1 - nmse) };
	}

	_psnr(mse) {
		if (mse === 0) return { psnr: this.maxPSNR, similarity: 1 };
		const psnr = 10 * Math.log10((255 * 255) / mse);
		const similarity = Math.max(0, Math.min(1, psnr / this.maxPSNR));
		return { psnr, similarity };
	}

	_stats(gray) {
		let mean = 0;
		for (let i = 0; i < gray.length; i++) mean += gray[i];
		mean /= gray.length;
		let varSum = 0;
		for (let i = 0; i < gray.length; i++) {
		const d = gray[i] - mean;
		varSum += d * d;
		}
		const variance = varSum / gray.length;
		return { mean, variance };
	}

	_cov(grayA, meanA, grayB, meanB) {
		let cov = 0;
		for (let i = 0; i < grayA.length; i++) {
		cov += (grayA[i] - meanA) * (grayB[i] - meanB);
		}
		return cov / grayA.length;
	}

	_ssim(grayA, grayB) {
		// Global SSIM approximation
		const { mean: muX, variance: sigmaX2 } = this._stats(grayA);
		const { mean: muY, variance: sigmaY2 } = this._stats(grayB);
		const sigmaXY = this._cov(grayA, muX, grayB, muY);

		const L = 255;
		const k1 = 0.01, k2 = 0.03;
		const C1 = (k1 * L) ** 2;
		const C2 = (k2 * L) ** 2;

		const numerator = (2 * muX * muY + C1) * (2 * sigmaXY + C2);
		const denominator = (muX*muX + muY*muY + C1) * (sigmaX2 + sigmaY2 + C2);
		let ssim = numerator / denominator;
		// Bound to [0..1]
		ssim = Math.max(0, Math.min(1, ssim));
		return { ssim, similarity: ssim };
	}

	_histIntersection(imageDataA, imageDataB, bins = 32) {
		const histA = new Float32Array(bins);
		const histB = new Float32Array(bins);
		const step = Math.floor(imageDataA.data.length / 4); // pixel count

		const push = (data, hist) => {
		for (let i = 0; i < data.length; i += 4) {
			// grayscale bin for simplicity (fast, robust)
			const g = 0.2126*data[i] + 0.7152*data[i+1] + 0.0722*data[i+2];
			const idx = Math.min(bins - 1, Math.floor(g / (256 / bins)));
			hist[idx]++;
		}
		};
		push(imageDataA.data, histA);
		push(imageDataB.data, histB);

		let intersection = 0, sumA = 0, sumB = 0;
		for (let i = 0; i < bins; i++) {
		intersection += Math.min(histA[i], histB[i]);
		sumA += histA[i];
		sumB += histB[i];
		}
		const denom = Math.max(sumA, sumB, 1);
		const similarity = intersection / denom; // [0..1]
		return { similarity, intersection, denom };
	}

	_dct2(block, N) {
		// Simple 2D DCT (naive, O(N^4) but N is small for pHash)
		const out = new Float32Array(N * N);
		const c = (x) => (x === 0 ? Math.sqrt(1/N) : Math.sqrt(2/N));
		for (let u = 0; u < N; u++) {
		for (let v = 0; v < N; v++) {
			let sum = 0;
			for (let x = 0; x < N; x++) {
			for (let y = 0; y < N; y++) {
				sum += block[x*N + y] *
				Math.cos(((2*x + 1) * u * Math.PI) / (2 * N)) *
				Math.cos(((2*y + 1) * v * Math.PI) / (2 * N));
			}
			}
			out[u*N + v] = c(u) * c(v) * sum;
		}
		}
		return out;
	}

	_pHash(gray, width, height) {
		// Resize to 32x32 then DCT, then 8x8 top-left
		const N = 32, K = 8;

		// Draw gray back to canvas to uniformly resize
		const tempCanvas = document.createElement('canvas');
		tempCanvas.width = N; tempCanvas.height = N;
		const ctx = tempCanvas.getContext('2d', { willReadFrequently: true });
		// Put grayscale into an ImageData so we can draw/scale via canvas
		const imgData = ctx.createImageData(width, height);
		for (let i = 0, p = 0; i < imgData.data.length; i += 4, p++) {
		const g = Math.max(0, Math.min(255, gray[p]));
		imgData.data[i] = g; imgData.data[i+1] = g; imgData.data[i+2] = g; imgData.data[i+3] = 255;
		}
		const srcCanvas = document.createElement('canvas');
		srcCanvas.width = width; srcCanvas.height = height;
		const sctx = srcCanvas.getContext('2d', { willReadFrequently: true });
		sctx.putImageData(imgData, 0, 0);
		ctx.drawImage(srcCanvas, 0, 0, N, N);
		const resized = ctx.getImageData(0, 0, N, N);

		// Build grayscale array for DCT input
		const block = new Float32Array(N * N);
		for (let i = 0, p = 0; i < resized.data.length; i += 4, p++) {
		block[p] = resized.data[i]; // already grayscale
		}

		const dct = this._dct2(block, N);
		// Compute mean of top-left 8x8 excluding DC (0,0)
		let sum = 0, count = 0;
		for (let u = 0; u < K; u++) {
		for (let v = 0; v < K; v++) {
			if (u === 0 && v === 0) continue;
			sum += dct[u*N + v]; count++;
		}
		}
		const mean = sum / Math.max(1, count);

		// Build 64-bit hash (as string of 64 bits)
		const bits = [];
		for (let u = 0; u < K; u++) {
		for (let v = 0; v < K; v++) {
			if (u === 0 && v === 0) continue;
			bits.push(dct[u*N + v] > mean ? 1 : 0);
		}
		}
		// bits length is 63 (since DC excluded). Pad to 64 by appending DC comparison to mean.
		bits.push(dct[0] > mean ? 1 : 0);

		return bits; // Array of 64 bits
	}

	_hamming(aBits, bBits) {
		const n = Math.min(aBits.length, bBits.length);
		let dist = 0;
		for (let i = 0; i < n; i++) if (aBits[i] !== bBits[i]) dist++;
		dist += Math.abs(aBits.length - bBits.length); // if lengths differ
		return dist;
	}

	/* ---------------------------- Image Comparison ---------------------------- */

	async _compareImages(a, b) {
		const srcA = await this._materializeSource(a);
		const srcB = await this._materializeSource(b);

		const dataA = this._drawToCanvas(srcA);
		const dataB = this._drawToCanvas(srcB);

		const grayA = this.grayscale ? this._toGrayscale(dataA) : { gray: null };
		const grayB = this.grayscale ? this._toGrayscale(dataB) : { gray: null };

		// MSE + PSNR
		const mseRes = this._mse(grayA.gray, grayB.gray);
		const psnrRes = this._psnr(mseRes.mse);

		// SSIM
		const ssimRes = this._ssim(grayA.gray, grayB.gray);

		// Histogram Intersection (uses original ImageData)
		const histRes = this._histIntersection(dataA, dataB);

		// pHash + Hamming
		const phA = this._pHash(grayA.gray, dataA.width, dataA.height);
		const phB = this._pHash(grayB.gray, dataB.width, dataB.height);
		const hamming = this._hamming(phA, phB);
		const phashSim = Math.max(0, 1 - (hamming / 64));

		const metrics = {
		mse: mseRes.similarity,
		psnr: psnrRes.similarity,
		ssim: ssimRes.similarity,
		hist: histRes.similarity,
		phash: phashSim
		};

		const similarity = this._weightedAggregate(metrics);
		return { similarity, metrics, kind: 'image' };
	}

	_weightedAggregate(metrics) {
		let wsum = 0, acc = 0;
		for (const key of Object.keys(this.weights)) {
		const w = this.weights[key] ?? 0;
		if (w <= 0) continue;
		const v = metrics[key] ?? 0;
		acc += w * v;
		wsum += w;
		}
		return wsum > 0 ? acc / wsum : 0;
	}

	/* ---------------------------- Video Comparison ---------------------------- */

	async _compareVideos(a, b) {
		const videoA = await this._materializeSource(a);
		const videoB = await this._materializeSource(b);

		const framesA = await this._sampleVideoFrames(videoA, this.videoFrameSamples);
		const framesB = await this._sampleVideoFrames(videoB, this.videoFrameSamples);

		const n = Math.min(framesA.length, framesB.length);
		let aggSimilarity = 0;
		const details = [];

		for (let i = 0; i < n; i++) {
		const { similarity, metrics } = await this._compareImages(framesA[i], framesB[i]);
		aggSimilarity += similarity;
		details.push(metrics);
		}

		const similarity = n > 0 ? aggSimilarity / n : 0;
		return { similarity, metricsPerFrame: details, kind: 'video' };
	}

	async _sampleVideoFrames(video, samples) {
		await this._awaitVideoReady(video);
		const duration = video.duration || 0;
		if (!duration || !isFinite(duration)) {
		// Fallback: single current frame
		return [video];
		}

		const timestamps = [];
		// Sample from 10% to 90% to avoid blank first/last frames
		const start = duration * 0.1, end = duration * 0.9;
		const step = (end - start) / (samples - 1);
		for (let i = 0; i < samples; i++) timestamps.push(start + i * step);

		const frames = [];
		for (const t of timestamps) {
		await this._seekVideo(video, t);
		const frameData = this._drawToCanvas(video);
		// Convert ImageData back into a pseudo-image source for reuse
		const frameCanvas = document.createElement('canvas');
		frameCanvas.width = frameData.width;
		frameCanvas.height = frameData.height;
		const ctx = frameCanvas.getContext('2d', { willReadFrequently: true });
		ctx.putImageData(frameData, 0, 0);
		frames.push(frameCanvas);
		}
		return frames;
	}

	_seekVideo(video, time) {
		return new Promise((resolve, reject) => {
		const onSeeked = () => {
			cleanup();
			resolve();
		};
		const onError = (e) => {
			cleanup();
			reject(e);
		};
		const cleanup = () => {
			video.removeEventListener('seeked', onSeeked);
			video.removeEventListener('error', onError);
		};
		video.addEventListener('seeked', onSeeked);
		video.addEventListener('error', onError);
		video.currentTime = Math.min(Math.max(0, time), video.duration || 0);
		});
	}

	/* ------------------------- Video-to-Image Fallback ------------------------ */

	async _compareVideoToImage(videoSrc, imageSrc) {
		const video = await this._materializeSource(videoSrc);
		const image = await this._materializeSource(imageSrc);
		const frames = await this._sampleVideoFrames(video, this.videoFrameSamples);

		let best = 0;
		let bestMetrics = null;
		for (const f of frames) {
		const { similarity, metrics } = await this._compareImages(f, image);
		if (similarity > best) {
			best = similarity;
			bestMetrics = metrics;
		}
		}
		return { similarity: best, metrics: bestMetrics || {}, kind: 'video-image' };
	}
};