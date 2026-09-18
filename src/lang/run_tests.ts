import { runTests, testSummary } from "./tests";

const filter = process.argv[2];
const all = runTests();
const rs = filter ? all.filter((r) => r.group === filter || r.name.includes(filter)) : all;
const s = testSummary(rs);

for (const r of rs.filter((r) => !r.ok)) console.log(`FAIL [${r.group}] ${r.name}\n      ${r.detail}\n`);
console.log(`${s.passed}/${s.total} tests passed${filter ? ` (filter: ${filter})` : ""}`);
process.exit(s.passed === s.total ? 0 : 1);
