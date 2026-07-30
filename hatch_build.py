"""Build hook: compile the React frontend into the wheel.

The UI is a Vite app under frontend/. Users installing the wheel must not need
Node, so the build machine runs `npm run build` here and the result is copied to
src/kaioken/_frontend/, which ships as package data (see assets.frontend_index).

Set KAIOKEN_SKIP_FRONTEND_BUILD=1 to reuse an existing frontend/dist -- useful in
CI where the frontend is built in a separate step.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

PACKAGE_FRONTEND = Path("src") / "kaioken" / "_frontend"


class FrontendBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version, build_data):
        root = Path(self.root)
        frontend = root / "frontend"
        dist = frontend / "dist"
        target = root / PACKAGE_FRONTEND

        if not frontend.is_dir():
            raise RuntimeError(f"frontend/ not found at {frontend}")

        skip = os.environ.get("KAIOKEN_SKIP_FRONTEND_BUILD") == "1"
        if skip and (dist / "index.html").is_file():
            self._log(f"reusing existing {dist} (KAIOKEN_SKIP_FRONTEND_BUILD=1)")
        else:
            if skip:
                raise RuntimeError(
                    f"KAIOKEN_SKIP_FRONTEND_BUILD=1 but {dist / 'index.html'} does not "
                    "exist; build the frontend first or unset the variable")
            self._npm_build(frontend)

        if not (dist / "index.html").is_file():
            raise RuntimeError(
                f"frontend build produced no {dist / 'index.html'}; refusing to "
                "package a wheel with no UI")

        # Rebuild from scratch: a stale file left from an older UI would ship
        # silently, and Vite's hashed asset names make that easy to miss.
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(dist, target)
        self._log(f"bundled {sum(1 for _ in target.rglob('*') if _.is_file())} "
                  f"frontend file(s) into {PACKAGE_FRONTEND}")

        build_data["artifacts"].append(f"/{PACKAGE_FRONTEND.as_posix()}/**")

    def _npm_build(self, frontend):
        npm = shutil.which("npm")
        if npm is None:
            raise RuntimeError(
                "npm not found on PATH. Install Node.js to build the wheel, or "
                "build frontend/dist yourself and set KAIOKEN_SKIP_FRONTEND_BUILD=1")

        # `npm ci` needs the lockfile and gives a reproducible tree; fall back to
        # `npm install` when there is none.
        install = ["ci"] if (frontend / "package-lock.json").is_file() else ["install"]
        for args in (install, ["run", "build"]):
            self._log(f"npm {' '.join(args)}")
            subprocess.run([npm, *args], cwd=frontend, check=True)

    def _log(self, message):
        print(f"[kaioken frontend] {message}", file=sys.stderr)
