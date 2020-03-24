{ pkgs ? import <nixpkgs> { }, ... }:

pkgs.stdenvNoCC.mkDerivation rec {
  name = "pycl-env";
  env = pkgs.buildEnv { name = name; paths = buildInputs; };

  buildInputs = let
    custom-python = let
      packageOverrides = self: super: {
        pyopencl = super.pyopencl.overridePythonAttrs(old: rec {
          buildInputs = with pkgs; [
            opencl-headers ocl-icd python37Packages.pybind11
            libGLU_combined
          ];
        # Enable OpenGL integration and fix build
          preBuild = ''
            python configure.py --cl-enable-gl
            export HOME=/tmp/pyopencl
          '';
        });
      };
    in pkgs.python3.override { inherit packageOverrides; };

    local-python = custom-python.withPackages (python-packages: with python-packages; [
      numpy
      scipy
      pyopencl setuptools
      pyopengl pyrr
      matplotlib
    ]);

  in [
    local-python
    pkgs.opencl-info
    pkgs.universal-ctags
  ];

  shellHook = ''
    export NIX_SHELL_NAME="${name}"
    export PYTHONPATH="$PWD/boltzgas:$PYTHONPATH"
    export PYOPENCL_COMPILER_OUTPUT=1
    export PYTHONDONTWRITEBYTECODE=1
  '';
}
