importpath=github.com/daaku/whispy
pkgname=$(basename "$importpath")
pkgver=$(git rev-list --count HEAD)
pkgrel=1
pkgdesc='Speech to text dictation for Wayland using whisper.cpp'
arch=('x86_64')
url="https://$importpath"
license=('MIT')
depends=(
  'glibc'             # libc, libm, libresolv
  'libgcc'            # libgcc_s
  'libstdc++'         # libstdc++
  'libgomp'           # OpenMP, used by ggml
  'vulkan-icd-loader' # libvulkan, used by the ggml vulkan backend
  'pipewire-audio'    # pw-record
  'wl-clipboard'      # wl-copy
  'wtype'             # wtype
  'xdg-utils'         # xdg-open, used in search mode
)
makedepends=(
  'go'
  'cmake'
  'git'
  'vulkan-headers'
  'shaderc'       # glslc, to compile the vulkan compute shaders
  'spirv-headers' # find_package(SPIRV-Headers CONFIG)
)
options=('!lto') # LTO of ggml is slow and very memory hungry

# whisper.cpp is not packaged separately, we build and bundle it here
_whisper_rev=1d549b3
_whisper_src=whisper.cpp

build() {
  cd ..

  if [[ ! -d $_whisper_src ]]; then
    git clone https://github.com/ggml-org/whisper.cpp "$_whisper_src"
  fi

  pushd "$_whisper_src"
  git checkout "$_whisper_rev"
  popd

  cmake -S "$_whisper_src" -B "$_whisper_src/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DWHISPER_BUILD_IS_DEV=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_VULKAN=ON \
    -DWHISPER_BUILD_EXAMPLES=OFF \
    -DWHISPER_BUILD_TESTS=OFF \
    -DWHISPER_BUILD_SERVER=OFF
  cmake --build "$_whisper_src/build" -j8 --target whisper parakeet

  # The libraries are installed into /usr/lib, which is in the default library
  # search path, so the binary does not need an rpath of its own. makepkg's
  # LDFLAGS do not reach the cgo link, so the hardening flags are passed here
  # explicitly. Note that --as-needed must not be used: it would drop
  # libggml-cpu and libggml-vulkan, which have no referenced symbols but do
  # register their backends when loaded.
  export CGO_ENABLED=1
  export CGO_LDFLAGS='-Wl,-z,relro,-z,now'
  go build -trimpath -o whispy .
}

package() {
  cd ..

  install -Dm755 whispy "$pkgdir/usr/bin/whispy"

  # Installs libwhisper, libparakeet and the libggml backends into /usr/lib,
  # while rewriting the rpath away from the build directory.
  DESTDIR="$pkgdir" cmake --install "$_whisper_src/build" --prefix /usr
  # Only the libraries are of interest here, not the headers, cmake or
  # pkgconfig files.
  rm -rf "$pkgdir/usr/include" "$pkgdir/usr/lib/cmake" "$pkgdir/usr/lib/pkgconfig"

  install -Dm644 license "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
}
