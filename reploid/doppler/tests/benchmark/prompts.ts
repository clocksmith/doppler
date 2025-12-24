/**
 * Standard Benchmark Prompts
 *
 * Fixed prompts for reproducible benchmarking.
 * Stored in repo (no network fetch during benchmark).
 *
 * @module tests/benchmark/prompts
 */

import type { Prompt, PromptCategory } from './types.js';

/**
 * Extra-small prompt: 6-10 tokens
 * Minimal completion for quick testing
 */
export const XS_PROMPT: Prompt = {
  name: 'xs',
  text: 'The color of the sky is',
  expectedTokenCount: { min: 6, max: 10 },
};

/**
 * Short prompt: 16-64 tokens
 * Simple completion task
 */
export const SHORT_PROMPT: Prompt = {
  name: 'short',
  text: 'The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.',
  expectedTokenCount: { min: 16, max: 64 },
};

/**
 * Medium prompt: 256-512 tokens
 * Technical explanation task
 */
export const MEDIUM_PROMPT: Prompt = {
  name: 'medium',
  text: `Explain the concept of recursion in computer science. Start with a simple definition, then provide examples in pseudocode showing both base cases and recursive cases. Discuss the trade-offs between recursive and iterative solutions, including stack usage, readability, and performance considerations.

Consider the following aspects:
1. What makes a problem suitable for recursion?
2. How does the call stack work during recursive execution?
3. What is tail recursion and why does it matter?
4. Common pitfalls when writing recursive functions.

Provide concrete examples such as factorial calculation, tree traversal, and the Fibonacci sequence. Compare the recursive implementations with their iterative counterparts.`,
  expectedTokenCount: { min: 256, max: 512 },
};

/**
 * Long prompt: ~2048 tokens
 * Complex multi-part task
 */
export const LONG_PROMPT: Prompt = {
  name: 'long',
  text: `You are a senior software architect reviewing a system design proposal. The proposal describes a distributed caching system with the following requirements:

## Functional Requirements

1. Support for key-value storage with string keys and arbitrary JSON values
2. TTL (time-to-live) support for automatic expiration
3. Support for cache invalidation patterns: exact key, prefix-based, and tag-based
4. Read-through and write-through caching modes
5. Support for distributed locks with configurable timeout
6. Pub/sub for cache invalidation events across nodes

## Non-Functional Requirements

1. P99 read latency under 5ms for cache hits
2. P99 write latency under 10ms
3. Support for 10,000 requests per second per node
4. Horizontal scalability to 100+ nodes
5. Data consistency guarantees during network partitions
6. Automatic failover with recovery time under 30 seconds

## Proposed Architecture

The team proposes using a consistent hashing ring for data distribution, with virtual nodes for better load balancing. Each physical node maintains:

- An in-memory LRU cache for hot data (configurable size, default 1GB)
- A write-ahead log for durability
- A gossip protocol for membership and failure detection
- A vector clock for conflict resolution

The replication strategy uses quorum-based reads and writes with configurable consistency levels (ONE, QUORUM, ALL).

## Your Task

Please review this proposal and provide feedback on:

1. Potential bottlenecks and single points of failure
2. Trade-offs in the consistency model
3. Operational concerns (monitoring, debugging, deployment)
4. Alternative approaches that might better meet the requirements
5. Specific implementation recommendations for critical components

Consider edge cases such as:
- Network partitions between data centers
- Thundering herd on cache misses
- Hot key distribution problems
- Memory pressure and eviction strategies
- Clock synchronization issues affecting TTL

Provide your analysis in a structured format with clear recommendations and justifications.`,
  expectedTokenCount: { min: 1800, max: 2200 },
};

/**
 * All standard prompts indexed by category
 */
export const PROMPTS: Record<PromptCategory, Prompt> = {
  xs: XS_PROMPT,
  short: SHORT_PROMPT,
  medium: MEDIUM_PROMPT,
  long: LONG_PROMPT,
};

/**
 * Get a prompt by category
 */
export function getPrompt(category: PromptCategory): Prompt {
  return PROMPTS[category];
}

/**
 * Get all prompt categories
 */
export function getPromptCategories(): PromptCategory[] {
  return ['xs', 'short', 'medium', 'long'];
}
