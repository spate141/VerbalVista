import requests
import importlib.metadata
from tabulate import tabulate


def get_installed_version(package_name):
    """
    Get the version of an installed package using importlib.metadata.
    """
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_latest_version_from_pypi(package_name):
    """
    Fetch the latest package version from PyPI.
    """
    response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
    if response.ok:
        return response.json()['info']['version']
    else:
        return None


def check_for_updates(requirements_file):
    """
    Check for package updates based on a requirements.txt file.
    """
    updates_available = []
    with open(requirements_file, "r") as file:
        for line in file.readlines():
            package_name = line.split("==")[0].strip()
            installed_version = get_installed_version(package_name)
            latest_version = get_latest_version_from_pypi(package_name)
            if installed_version and latest_version and installed_version != latest_version:
                updates_available.append([package_name, installed_version, latest_version])
    return updates_available


def main():
    requirements_file = "requirements.txt"
    updates = check_for_updates(requirements_file)
    print(tabulate(updates, headers=['Package', 'Installed Version', 'Latest Version'], tablefmt='fancy_grid'))


if __name__ == '__main__':
    main()

