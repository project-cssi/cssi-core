#!/usr/bin/env bash

set -u -e -o pipefail

# Used for production releases.
# Args: release_type: ["major", "minor", "patch", "premajor", "preminor", "prepatch", "prenext", "dev", "post"]
#       release_level: ["alpha", "beta", "candidate", "final"]

RELEASE_TYPE=$1
RELEASE_LEVEL=$2

echo "======================================================"
echo "           Releasing CSSI for Production\n"
echo "              Release Type: $RELEASE_TYPE"
echo "              Release Level: $RELEASE_LEVEL"
echo "======================================================"

# Construct the new version by executing the `construct_release_version()` function
# in `version.py` script inside the `cssi` package.
NEW_VERSION=$(python -c "from cssi.version import construct_release_version; print(construct_release_version('${RELEASE_TYPE}', '${RELEASE_LEVEL}'))")

# Tag the release
git tag -a ${NEW_VERSION} -m "release: cut the v$NEW_VERSION release"
git push origin ${NEW_VERSION}

# Generate the changelog
# Format -o = output file
#        -s = commit style {angular,atom,basic}
#        -t = template {angular, keepachangelog}
#        REPOSITORY = directory that contains the .git folder
git-changelog -o CHANGELOG.md -s angular -t angular .

# Stage and create a release commit with CHANGELOG.md and VERSION.txt files
git add CHANGELOG.md cssi/VERSION.txt
git commit -m "release: cut the v$NEW_VERSION release :tada:"

echo "Successfully released a new version of CSSI: v$NEW_VERSION"
