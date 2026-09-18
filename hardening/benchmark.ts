/** Export TENSA-generated workloads for an explicit CPU-vs-GPU benchmark.
 * npx tsx hardening/benchmark.ts
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { compile } from "../src/lang/analyze";
import { emitTorch } from "../src/lang/emit_torch";
import { VIT } from "../src/lang/challenges/research";
const directory = "hardening/.m4-validation/benchmark";
mkdirSync(directory, { recursive: true });
const workloads = [
  { id: "tiny-m4-vit", code: VIT, batch: 2, height: 4, width: 6, tokens: 7, channels: 8, depth: 1 },
  ...[
    { id: "medium-vit", batch: 8, height: 128, width: 128, tokens: 65, channels: 128, depth: 2 },
    { id: "larger-vit", batch: 8, height: 224, width: 224, tokens: 197, channels: 256, depth: 4 },
  ].map(w => ({ ...w, code: `dim B
dim H = ${w.height}
dim W = ${w.width}
dim P = 16
dim N = (H/P)*(W/P)
dim D = ${w.channels}
block EncoderBlock(x: Tensor[B,N+1,D]) -> Tensor[B,N+1,D] {
  residual { layernorm ; attention(heads: 4) }
  residual { layernorm ; linear(D*4) ; gelu ; linear(D) }
}
model ViT(x: Image[B,3,H,W]) -> Tensor[B,32] {
  param cls: Tensor[1,1,D] init: normal
  let grid = reshape(x,[B,3,H/P,P,W/P,P])
  let order = transpose(transpose(transpose(grid,1,2),2,4),3,4)
  let patches = reshape(order,[B,N,3*P*P]) |> linear(D)
  let token = mean(patches,axis: 1,keep: true)*0.0 + cls
  let tokens = concat(token,patches,axis: 1) |> positional(max: N+1)
  tokens
  for ${w.depth}: EncoderBlock()
  let norm = layernorm
  return norm[:,0,:] |> linear(32)
}` })),
];
for (const w of workloads) {
  const c = compile(w.code, w.id);
  if (!c.ok) throw new Error(`${w.id}: ${c.errors.map(d => d.message).join("; ")}`);
  writeFileSync(`${directory}/${w.id}.tensa`, w.code);
  writeFileSync(`${directory}/${w.id}.py`, emitTorch(c.mod));
}
writeFileSync(`${directory}/manifest.json`, JSON.stringify(workloads.map(({code: _code, ...w}) => w), null, 2));
console.log(`Exported ${workloads.length} TENSA workloads to ${directory}`);
