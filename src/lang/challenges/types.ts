/**
 * TENSA hardening — challenge corpus types.
 *
 * A Challenge is a program plus the layers of the protocol's §2.2 checklist that
 * can be asserted mechanically, plus its evil twins (§2.3).  The driver turns
 * every populated expectation into one test, and every twin into one test.
 */
import { IRModule } from "../ir";
import { Effect } from "../types";

export type Tier = 1 | 2 | 3 | 4 | 5 | 6;

export interface EvilTwin {
  id: string;
  /** one line: what was changed relative to the parent program */
  mutates: string;
  code: string;
  /** diagnostic codes that MUST fire */
  expectCodes: string[];
  /** codes that must NOT fire (guards against false certainty / cascades) */
  forbidCodes?: string[];
  /** 1-based source line the primary diagnostic must point at */
  line?: number;
}

export interface RunExpectation {
  dims?: Record<string, number>;
  steps?: number;
  /** model name -> substring that must appear in the runtime output shape */
  outputs?: Record<string, string>;
  /** every trainable parameter receives a gradient update */
  gradAll?: boolean;
  /** runtime must refuse (errors non-empty); used for carried-constraint violations */
  refuses?: boolean;
  /** substring that must appear in the first runtime error when `refuses` is set */
  errorContains?: string;
}

/** A structural invariant on the IR that no declarative field expresses. */
export interface CustomCheck {
  name: string;
  /** throws on violation; returns a one-line detail on success */
  check: (mod: IRModule) => string;
}

export interface Expectation {
  /** compile must succeed with no errors (default true) */
  ok?: boolean;
  /** warning codes that are expected; any other warning is a failure when set */
  warnCodes?: string[];
  paramCount?: number;
  paramTables?: number;
  /** exact owner strings that must exist in mod.params */
  paramOwners?: string[];
  sharedGroups?: number;
  stateSlots?: number;
  /** effects that must appear on at least one IR node */
  effects?: Effect[];
  /** ops that must appear somewhere in the IR (including regions) */
  irOps?: string[];
  irForbidOps?: string[];
  /** constraint statuses that must be present */
  constraints?: ("proved" | "assumed" | "failed")[];
  inspectContains?: string[];
  emitContains?: string[];
  emitForbids?: string[];
  run?: RunExpectation;
  checkpointKinds?: string[];
  /** model name -> exact static output shape text, e.g. "[B, 8, ⌊H/2⌋]" (§5 algebra) */
  outputShape?: Record<string, string>;
  /** structural invariants asserted directly on the IR */
  custom?: CustomCheck[];
}

export interface Challenge {
  id: string;
  title: string;
  /** protocol section(s), e.g. "§7.1" */
  section: string;
  tier?: Tier;
  code: string;
  expect: Expectation;
  twins: EvilTwin[];
  /** path under hardening/records/ when a durable record exists */
  record?: string;
}
