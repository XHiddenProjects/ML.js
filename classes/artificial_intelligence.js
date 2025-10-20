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