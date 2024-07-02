import os
import subprocess
import sys


def run_command(command, cwd=None):
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        print(result.stdout)
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Command '{command}' failed with error: {e.stderr}")
        sys.exit(1)


def main():
    # CMake build directory
    build_dir = "/home/nurullah/Masaüstü/Stella_test/build"

    # Create the build directory if it does not exist
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    # Run CMake
    cmake_command = f"cmake -S . -B {build_dir}"
    run_command(cmake_command)

    # Build the project
    build_command = f"cmake --build {build_dir}"
    run_command(build_command)

    # Run the executable with arguments
    executable = os.path.join(build_dir, "slam_example")
    vocab_file = "/home/nurullah/Masaüstü/Stella_test/orb_vocab.fbow"
    config_file = "/home/nurullah/Masaüstü/Stella_test/config.yaml"
    images_folder = "/home/nurullah/Masaüstü/tekno_server/tekno_server/frames1/"

    run_command(f"{executable} {vocab_file} {config_file} {images_folder}")


if __name__ == "__main__":
    main()
