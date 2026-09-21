importpath=github.com/daaku/whispy
pkgname=$(basename "$importpath")
pkgver=$(git rev-list --count HEAD)
pkgrel=1
pkgdesc='Speech to text dictation for Wayland using Parakeet and OpenVINO'
arch=('x86_64')
url="https://$importpath"
license=('MIT')
depends=(
  'glibc'          # libc, libm, libresolv
  'libgcc'         # libgcc_s
  'libstdc++'      # libstdc++
  'openvino'       # libopenvino_c, runs parakeet
  'pipewire-audio' # pw-record
  'wl-clipboard'   # wl-copy
  'wtype'          # wtype
  'xdg-utils'      # xdg-open, used in search mode
)
makedepends=('go')

build() {
  cd ..

  # makepkg's LDFLAGS do not reach the cgo link, so the hardening flags are
  # passed here explicitly.
  export CGO_ENABLED=1
  export CGO_LDFLAGS='-Wl,-z,relro,-z,now'
  # The vad's matrix kernels are only built with the simd experiment. Setting
  # GOEXPERIMENT replaces whatever this go build defaults to, and this one
  # defaults to nodwarf5: dropping that turns DWARF5 back on, which makes
  # debugedit log "Unsupported .debug_line directory 0 path DW_FORM_0x8". Add
  # simd without losing the default.
  goexp="$(go env GOEXPERIMENT)"
  export GOEXPERIMENT="${goexp:+$goexp,}simd"
  go build -trimpath -o whispy .
}

package() {
  cd ..
  install -Dm755 whispy "$pkgdir/usr/bin/whispy"
  install -Dm644 license "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
}
