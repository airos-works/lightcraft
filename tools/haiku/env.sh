# Cross-compilation environment for air/OS (Haiku x86_64). Source it:
#   . tools/haiku/env.sh
# then build with tools/haiku/build.sh. Expects the toolchain under
# /mnt/HaikuWork/toolchains (rustup stable with rust-src, the Haiku sysroot)
# and the Haiku cross gcc of the x399 build tree.
LC_HAIKU_TOOLS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export RUSTUP_HOME=${RUSTUP_HOME:-/mnt/HaikuWork/toolchains/rust/rustup}
export CARGO_HOME=${CARGO_HOME:-/mnt/HaikuWork/toolchains/rust/cargo}
export PATH="$CARGO_HOME/bin:$PATH"
export HAIKU_SYSROOT=${HAIKU_SYSROOT:-/mnt/HaikuWork/toolchains/haiku-sysroot-x86_64}
export HAIKU_CROSS=${HAIKU_CROSS:-/mnt/HaikuWork/x399/build/x86_64/cross-tools-x86_64/bin/x86_64-unknown-haiku}
# x86_64-unknown-haiku is a tier-3 target: std is built from rust-src.
export RUSTC_BOOTSTRAP=1
export CARGO_TARGET_X86_64_UNKNOWN_HAIKU_LINKER="$LC_HAIKU_TOOLS/haiku-cc"
export CC_x86_64_unknown_haiku="$LC_HAIKU_TOOLS/haiku-cc"
export CXX_x86_64_unknown_haiku="$LC_HAIKU_TOOLS/haiku-c++"
export AR_x86_64_unknown_haiku="$HAIKU_CROSS-ar"
export RANLIB_x86_64_unknown_haiku="$HAIKU_CROSS-ranlib"
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$(cd -- "$LC_HAIKU_TOOLS/../.." && pwd)/target}
