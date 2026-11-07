#!/usr/bin/env bash
# Build the Linux frozen add-on, its archive and the sources the release attaches,
# inside an AlmaLinux 8 container.
#
# Copyright (C) 2026 Subtitld contributors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# The bundle carries the build interpreter's libpython and a compiled espeak-ng,
# so it runs on any glibc at least as new as the build's. The image
# (tools/linux-build.Dockerfile) is AlmaLinux 8, glibc 2.28, as for the other
# Subtitld add-ons (addon-phonon2), with the distribution's Python 3.12.
#
#     tools/build_linux.sh
#
# writes, into the repository, owned by you:
#     paradee-<version>-linux-x86_64.zip                  (+ .sha256)
#     paradee-<version>-libquadmath-source.tar.gz         (+ .sha256)  LGPL-2.1, numpy's
#     paradee-<version>-source.tar.gz                     (+ .sha256)  GPL-3.0 Corresponding Source
#
# The Corresponding Source archives the add-on at git HEAD: commit first.
# Needs Docker and the network.

set -euo pipefail

BUILD_IMAGE="${BUILD_IMAGE:-subtitld-paradee-build:almalinux8-py312}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(sed -n 's/^__version__ = .\(.*\).$/\1/p' "$ROOT/paradee_addon/__init__.py")"
ARCHIVE="paradee-${VERSION}-linux-x86_64.zip"
QUADMATH_SRC="paradee-${VERSION}-libquadmath-source.tar.gz"
SOURCE="paradee-${VERSION}-source.tar.gz"
# pip's cache and the downloaded source packages survive between builds.
PIP_CACHE="${PIP_CACHE:-$HOME/.cache/subtitld-paradee-build-pip}"
SOURCES_CACHE="${SOURCES_CACHE:-$HOME/.cache/subtitld-paradee-build-sources}"
mkdir -p "$PIP_CACHE" "$SOURCES_CACHE"

if [ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]; then
    echo "warning: uncommitted changes; the Corresponding Source archives git HEAD, not them" >&2
fi

# No build context: the Dockerfile copies nothing in.
docker build -q -t "$BUILD_IMAGE" - < "$ROOT/tools/linux-build.Dockerfile" >/dev/null

docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$ROOT:/src" \
    -v "$PIP_CACHE:/tmp/.cache/pip" \
    -v "$SOURCES_CACHE:/tmp/.cache/sources" \
    -w /src \
    "$BUILD_IMAGE" \
    bash -euo pipefail -c "
        command -v nm >/dev/null || { echo 'nm is required for the libexpat check' >&2; exit 1; }
        git config --global --add safe.directory /src
        # The distribution's interpreter: /usr/local/bin/python3.12 is the image's static one.
        /usr/bin/python3.12 -m venv /tmp/build-venv
        . /tmp/build-venv/bin/activate
        pip install -q --upgrade pip
        pip install -q -c constraints.txt '.[build]'

        # A failed build must not leave older outputs that look current.
        rm -rf build/pyinstaller dist release-staging '$ARCHIVE' '$ARCHIVE.sha256' \\
            '$QUADMATH_SRC' '$QUADMATH_SRC.sha256' '$SOURCE' '$SOURCE.sha256'

        PYTHON=python tools/build_espeak.sh build/espeak-ng
        # Stripping (pyinstaller.spec) caches the stripped copies in PyInstaller's
        # config folder; /tmp/.cache belongs to root here, made by the pip mount.
        PYINSTALLER_CONFIG_DIR=/tmp/pyinstaller-config \\
            pyinstaller pyinstaller.spec --noconfirm --log-level WARN --workpath build/pyinstaller
        dist/paradee-addon/paradee-addon --self-test

        # Every LGPL library in the bundle must be a build whose source is attached.
        python tools/build_libquadmath_source.py --check-bundle dist/paradee-addon

        mkdir -p release-staging/bin
        cp -R dist/paradee-addon/. release-staging/bin/
        cp manifest.json LICENSE README.md release-staging/
        python tools/collect_licences.py --bundle dist/paradee-addon \\
            --out release-staging/THIRD_PARTY_LICENSES --needs-source /tmp/sources.txt
        python tools/make_archive.py release-staging '$ARCHIVE'
        sha256sum '$ARCHIVE' > '$ARCHIVE.sha256'

        python tools/build_libquadmath_source.py --srpm-dir /tmp/.cache/sources --out '$QUADMATH_SRC'
        python tools/build_source_tarball.py --out '$SOURCE'
        sha256sum '$QUADMATH_SRC' > '$QUADMATH_SRC.sha256'
        sha256sum '$SOURCE' > '$SOURCE.sha256'
    "

echo "built $ARCHIVE and the source tarballs"
cat "$ROOT/$ARCHIVE.sha256" "$ROOT/$QUADMATH_SRC.sha256" "$ROOT/$SOURCE.sha256"
