import { homedir } from "node:os";
import { join } from "node:path";

// BGE-small-en-v1.5 requires a query prefix for asymmetric search.
// Passage (memory content) gets no prefix - the model was trained
// to encode passages without instruction.
const QUERY_PREFIX =
  "Represent this sentence for searching relevant passages: ";

/**
 * Writable directory for the downloaded model files.
 *
 * transformers.js defaults its filesystem cache to a `.cache` folder *inside*
 * its own package directory (`node_modules/@huggingface/transformers/.cache`).
 * That breaks any read-only install - a global `bun install -g`, and Nix store
 * paths in particular - and gets wiped on every reinstall. Anchor the cache in
 * the user's XDG cache instead so it survives upgrades and works from an
 * immutable store path.
 *
 *   THATCH_MODEL_CACHE  explicit override (highest priority)
 *   XDG_CACHE_HOME      standard base dir -> $XDG_CACHE_HOME/thatch/models
 *   default             ~/.cache/thatch/models
 */
export function modelCacheDir(): string {
  const explicit = process.env.THATCH_MODEL_CACHE;
  if (explicit) return explicit;
  const cacheHome = process.env.XDG_CACHE_HOME ?? join(homedir(), ".cache");
  return join(cacheHome, "thatch", "models");
}

/**
 * Interface for an embedding model so tests can supply a mock.
 * `name` is the tag stored alongside each entry so a future reader can tell
 * which model produced a vector; recall discriminates spaces by dimension,
 * not by tag.
 */
export interface EmbeddingModel {
  readonly loaded: boolean;
  readonly name: string;
  load(): Promise<void>;
  queryEmbed(text: string): Promise<Float32Array>;
  passageEmbed(text: string): Promise<Float32Array>;
}

/**
 * Factory for creating a Hugging Face feature-extraction pipeline.
 * Injected into BgeEmbeddingModel for testability.
 */
export type PipelineFactory = (modelName: string) => Promise<any>;

const defaultPipelineFactory: PipelineFactory = async (modelName) => {
  const { pipeline, env } = await import("@huggingface/transformers");
  // Redirect the model cache out of the (possibly read-only) package tree.
  env.cacheDir = modelCacheDir();
  return pipeline("feature-extraction", modelName);
};

/**
 * Lazy-loads an embedding model via @huggingface/transformers.
 * Model files (~34 MB for the default) are downloaded once and cached by HF Hub.
 */
export class BgeEmbeddingModel implements EmbeddingModel {
  #modelName: string;
  #pipelineFactory: PipelineFactory;
  #pipe: any = null;
  #loading: Promise<void> | null = null;

  constructor(modelName = "Xenova/bge-small-en-v1.5", pipelineFactory?: PipelineFactory) {
    this.#modelName = modelName;
    this.#pipelineFactory = pipelineFactory ?? defaultPipelineFactory;
  }

  get loaded(): boolean {
    return this.#pipe !== null;
  }

  get name(): string {
    return this.#modelName;
  }

  // Memoizes the in-flight load so concurrent embed calls share one model
  // initialization. A failed load clears the memo so a later call can retry.
  async load(): Promise<void> {
    if (this.#pipe) return;
    this.#loading ??= (async () => {
      this.#pipe = await this.#pipelineFactory(this.#modelName);
    })().catch((err) => {
      this.#loading = null;
      throw err;
    });
    await this.#loading;
  }

  async queryEmbed(text: string): Promise<Float32Array> {
    return this.#embed(QUERY_PREFIX + text);
  }

  async passageEmbed(text: string): Promise<Float32Array> {
    return this.#embed(text);
  }

  async #embed(text: string): Promise<Float32Array> {
    await this.load();
    const output = await this.#pipe(text, {
      pooling: "mean",
      normalize: true,
    });
    return output.data as Float32Array;
  }
}
