import { copyFileSync, existsSync, mkdirSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { spawnSync } from 'node:child_process'

const clientRoot = resolve(import.meta.dirname, '..')
const crateRoot = join(clientRoot, 'wasm', 'aruco-core')
const wasmTarget = 'wasm32-unknown-unknown'
const artifactPath = join(
  crateRoot,
  'target',
  wasmTarget,
  'release',
  'aruco_core.wasm',
)
const outputPath = join(clientRoot, 'src', 'wasm', 'aruco_core.wasm')

function run(command, args, env = process.env) {
  const result = spawnSync(command, args, {
    cwd: clientRoot,
    env,
    encoding: 'utf8',
    stdio: 'inherit',
  })

  if (result.status !== 0) {
    process.exit(result.status ?? 1)
  }
}

const targetList = spawnSync('rustup', ['target', 'list', '--installed'], {
  cwd: clientRoot,
  encoding: 'utf8',
})

if (targetList.status !== 0) {
  process.exit(targetList.status ?? 1)
}

if (!targetList.stdout.includes(wasmTarget)) {
  run('rustup', ['target', 'add', wasmTarget])
}

const cargoEnv = {
  ...process.env,
  RUSTFLAGS: [process.env.RUSTFLAGS, '-C target-feature=+simd128']
    .filter(Boolean)
    .join(' '),
}

run(
  'cargo',
  ['build', '--manifest-path', join(crateRoot, 'Cargo.toml'), '--release', '--target', wasmTarget],
  cargoEnv,
)

if (!existsSync(artifactPath)) {
  console.error('WASM artifact was not produced at', artifactPath)
  process.exit(1)
}

mkdirSync(dirname(outputPath), { recursive: true })
copyFileSync(artifactPath, outputPath)
