# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Final Releases:
#   X(.Y)*
#   Ex: 0.9
#       0.9.1
#       0.9.2
# Pre-releases
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y.Z   # For bug-fix releases
#
# Post-releases:
#   X.YaN.postM   # Post-release of an alpha release
#   X.YbN.postM   # Post-release of a beta release
#   X.YrcN.postM  # Post-release of a release candidate
#
# Developmental releases
#   X.YaN.devM       # Developmental release of an alpha release
#   X.YbN.devM       # Developmental release of a beta release
#   X.YrcN.devM      # Developmental release of a release candidate
#   X.Y.postN.devM   # Developmental release of a post-release

_version = (0, 1, 0, "alpha", 1, 0, 0)


def _construct_version(major, minor, patch, level, pre_identifier, dev_identifier, post_identifier):
    """Construct a PEP0440 compatible version number to be set to __version__"""
    assert level in ["alpha", "beta", "candidate", "final"]

    version = "{0}.{1}".format(major, minor)
    if patch:
        version += ".{0}".format(patch)

    if level == "final":
        if post_identifier:
            version += ".{0}{1}".format("post", post_identifier)
        if dev_identifier:
            version += ".{0}{1}".format("dev", dev_identifier)
    else:
        level_short = {"alpha": "a", "beta": "b", "candidate": "rc"}[level]
        version += "{0}{1}".format(level_short, pre_identifier)
        if post_identifier:
            version += ".{0}{1}".format("post", post_identifier)
        if dev_identifier:
            version += ".{0}{1}".format("dev", dev_identifier)
    return version


def construct_release_version(release_type, release_level="final"):
    assert release_type in ["major", "minor", "patch", "premajor", "preminor", "prepatch", "prenext", "dev", "post"]
    assert release_level in ["alpha", "beta", "candidate", "final"]

    major, minor, patch, level, pre_identifier, dev_identifier, post_identifier = _version

    version = "{0}.{1}".format(major, minor)
    if release_type == "major":
        major += 1
        version = (major, 0, 0, release_level, 0, 0, 0)
    elif release_type == "minor":
        minor += 1
        version = (major, minor, 0, release_level, 0, 0, 0)
    elif release_type == "patch":
        patch += 1
        version = (major, minor, patch, release_level, 0, 0, 0)
    elif release_type == "premajor":
        major += 1
        version = (major, 0, 0, release_level, 1, 0, 0)
    elif release_type == "preminor":
        minor += 1
        version = (major, minor, 0, release_level, 1, 0, 0)
    elif release_type == "prepatch":
        patch += 1
        version = (major, minor, patch, release_level, 1, 0, 0)
    elif release_type == "prenext":
        pre_identifier += 1
        version = (major, minor, patch, release_level, pre_identifier, 0, 0)
    elif release_type == "dev":
        dev_identifier += 1
        version = (major, minor, patch, release_level, pre_identifier, dev_identifier, post_identifier)
    elif release_type == "post":
        post_identifier += 1
        version = (major, minor, patch, release_level, pre_identifier, dev_identifier, post_identifier)

    return _construct_version(*version)


__version__ = _construct_version(*_version)
