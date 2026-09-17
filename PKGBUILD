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
  'openvino'       # libopenvino_c, runs parakeet and the silero vad
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
  go build -trimpath -o whispy .
}

package() {
  cd ..
  install -Dm755 whispy "$pkgdir/usr/bin/whispy"
  install -Dm644 license "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
}
